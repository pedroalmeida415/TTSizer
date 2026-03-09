import torch
import soundfile as sf
import librosa
import numpy as np
import yaml
from tqdm.auto import tqdm
from typing import Dict, Any
import torch.nn as nn
from ml_collections import ConfigDict
from pathlib import Path

from ttsizer.models.bs_roformer import MelBandRoformer
from ttsizer.utils.vocal_utils import demix
from ttsizer.utils.logger import get_logger

import warnings
warnings.filterwarnings("ignore")

logger = get_logger("vocals_extractor")

class VocalsExtractor:
    """Separates vocals from audio files using a pre-trained model.
    
    Uses a MelBandRoformer model to isolate vocal tracks from mixed audio.
    Supports both single and multi-GPU processing, with configurable output formats
    and sample rates. Handles audio resampling and channel conversion as needed.
    """
    def __init__(self, config: Dict[str, Any]):
        """Initializes the VocalsExtractor with configuration settings.

        Args:
            config: A dictionary containing settings for vocal extraction,
                    typically from the 'vocals_extractor' section of config.yaml.
        """
        self.model_path = Path(config['model_path'])
        self.model_config_path = Path(config['model_config_path'])
        
        self.model_type = config["model_type"]
        self.use_gpu = config["use_gpu"]
        self.gpu_ids = config.get("gpu_ids", [0])
        self.out_fmt = config["output_format"].lower()
        self.pcm_type = config["output_pcm_type"]
        self.skip_existing = config.get("skip_if_output_exists", True)
        
        self.out_dir = None
        self.model_cfg = None
        self.model = None
        self.device = None
        self.target_sr = 44100

        if self.out_fmt not in ['wav', 'flac']:
            self.out_fmt = "flac"

        self._load_model()
        self._setup_device()

    def _load_model(self):
        """Loads the vocal separation model and its configuration."""
        with open(self.model_config_path) as f:
            self.model_cfg = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
        
        self.target_sr = self.model_cfg.audio.get('sample_rate', 44100)
            
        model_params = dict(self.model_cfg.model)
        if self.model_type == 'mel_band_roformer':
            self.model = MelBandRoformer(**model_params)
        else:
            raise ValueError(f"Unsupported model: {self.model_type}")

        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def _setup_device(self):
        """Configures the device (CPU or GPU) for model inference."""
        if self.use_gpu and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            valid_ids = [i for i in self.gpu_ids if 0 <= i < num_gpus]
            if not valid_ids:
                self.device = torch.device('cpu')
            elif len(valid_ids) == 1:
                self.device = torch.device(f'cuda:{valid_ids[0]}')
            else:
                self.device = torch.device(f'cuda:{valid_ids[0]}') 
                self.model = nn.DataParallel(self.model, device_ids=valid_ids) 
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)

    def _prepare_audio(self, mix: np.ndarray) -> np.ndarray:
        """Prepares the input audio mix to match model's expected channel format.

        Args:
            mix: NumPy array representing the input audio waveform.

        Returns:
            NumPy array of the audio waveform, adjusted for channel requirements.
        """
        if mix.ndim == 1:
            mix = np.expand_dims(mix, axis=0) 
        
        ch = self.model_cfg.audio.get('num_channels', 1)
        
        if mix.shape[0] == 1 and ch == 2:
            mix = np.concatenate([mix, mix], axis=0) 
        elif mix.shape[0] > ch:
            mix = mix[:ch, :]
        return mix

    def _process_file(self, path: Path, out_path: Path):
        """Processes a single audio file to extract vocals.

        Args:
            path: Path to the input audio file.
        """
        mix, sr = librosa.load(path, sr=None, mono=False)
        if sr != self.target_sr:
            if mix.ndim == 1: mix = np.expand_dims(mix, axis=0)
            mix = librosa.resample(mix, orig_sr=sr, target_sr=self.target_sr, res_type='kaiser_best')
        
        mix = self._prepare_audio(mix)
        mix_tensor = torch.tensor(mix, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            stems = demix(self.model_cfg, self.model, mix_tensor, self.device,
                         model_type=self.model_type, pbar=True)
        
        target = self.model_cfg.training.get('target_instrument', 'vocals')
        vocal_key = target if target in stems else 'vocals' if 'vocals' in stems else None

        if not vocal_key:
            return

        vocals = stems[vocal_key]


        subtype = self.pcm_type
        if self.out_fmt == 'wav' and subtype not in ['PCM_16', 'PCM_24', 'FLOAT']:
            subtype = 'FLOAT'
        elif self.out_fmt == 'flac' and subtype not in ['PCM_16', 'PCM_24']:
            subtype = 'PCM_16' 

        data = vocals.T if vocals.ndim > 1 else vocals[:, np.newaxis]
        sf.write(str(out_path), data, self.target_sr, subtype=subtype)

    def process_directory(self, in_dir: Path, out_dir: Path):
        """Extracts vocals from all supported audio files in a directory.

        Replicates the input directory structure (partially) for outputs.

        Args:
            in_dir: Path to the directory containing audio files.
            out_dir: Path to the directory where extracted vocals will be saved.
        """
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing: {in_dir}")
        logger.info(f"Output: {self.out_dir}")

        files = []
        for ext in ['*.flac', '*.wav']:
            files.extend(in_dir.rglob(ext))
        
        files = sorted(set(files))

        if not files:
            logger.warning("No audio files found")
            return
            
        logger.info(f"Found {len(files)} files")

        skipped = 0
        processed = 0

        for path in tqdm(files, desc="Extracting vocals", unit="file", position=0):
            rel_path = path.relative_to(in_dir)
            out_path = self.out_dir / rel_path.with_name(f"{path.stem}_vocals.{self.out_fmt}")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if self.skip_existing and out_path.exists():
                skipped += 1
                continue
            
            self._process_file(path, out_path)
            processed += 1

        logger.info(f"Done: {processed} processed, {skipped} skipped")


if __name__ == '__main__':
    with open("configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
        
    base = Path(cfg["project_setup"]["output_base_dir"]) / cfg["project_setup"]["series_name"]
    in_dir = base / cfg["audio_extractor"]["output_folder"]
    out_dir = base / cfg["vocals_extractor"]["output_folder"]
    
    print(f"Extracting vocals from {in_dir}")
    extractor = VocalsExtractor(config=cfg["vocals_extractor"])
    extractor.process_directory(in_dir, out_dir) 