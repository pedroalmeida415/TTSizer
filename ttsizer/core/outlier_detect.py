import warnings
warnings.filterwarnings("ignore")

import torch
import soundfile as sf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import torchaudio.functional as F
import wespeaker
import shutil
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from ttsizer.utils.logger import get_logger

logger = get_logger("outlier_detector")


class OutlierDetector:
    """Detects and moves audio clips that deviate from a target speaker's voice profile.

    This class uses speaker embeddings to create a voice profile (centroid) for each
    target speaker. It then compares individual audio clips against this profile
    and categorizes them as 'definite', 'uncertain', or 'outlier' based on cosine
    similarity thresholds. Files are moved to corresponding subdirectories.
    """

    def __init__(self, global_config: Dict[str, Any], outlier_config: Dict[str, Any]):
        """Initializes the OutlierDetector with global and detector-specific configurations.

        Args:
            global_config: Dictionary containing global project setup information.
            outlier_config: Dictionary containing configuration specific to the outlier detector.
        """
        self.target_speakers = global_config['project_setup']['target_speaker_labels']
        
        self.sr = outlier_config["target_sample_rate"]
        self.use_gpu = outlier_config["use_gpu"]
        self.model_path = outlier_config['embedding_model_path']
        
        self.min_duration = outlier_config["min_clip_duration_seconds"]
        self.refine_pct = outlier_config["centroid_refinement_percentile"]
        self.min_refine = outlier_config["min_segments_for_refinement"]
        self.min_profile = outlier_config["min_segments_for_master_profile"]
        self.def_thresh = outlier_config["outlier_threshold_definite"]
        self.unc_thresh = outlier_config["outlier_threshold_uncertain"]
    
        self.def_dir = "definite"
        self.unc_dir = "uncertain"
        self.outlier_dir = "outliers"
        
        self.model = self._load_model()

    def _load_model(self):
        """Loads the speaker embedding model using wespeaker."""
        model = wespeaker.load_model_local(str(self.model_path))
        model.set_device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        return model

    def _get_embedding(self, path: Path) -> Optional[np.ndarray]:
        """Extracts a speaker embedding from an audio file.

        Loads audio with soundfile, converts to mono if needed, and passes the
        PCM tensor directly to wespeaker (avoiding temp file round-trips for
        resampled audio).

        Args:
            path: Path to the audio file.

        Returns:
            A NumPy array of the speaker embedding, or None if processing fails
            or the clip is too short.
        """
        info = sf.info(str(path))
        if info.duration < self.min_duration:
            return None
        
        sig, sr = sf.read(str(path), dtype='float32')
        sig = torch.from_numpy(np.asarray(sig))
        
        # if > 1 channel, convert to mono
        if sig.ndim > 1 and sig.shape[-1] > 1:
            sig = torch.mean(sig, dim=-1)
        if sig.ndim == 1:
            sig = sig.unsqueeze(0)

        # Pass PCM tensor directly to wespeaker — handles resampling + fbank internally
        emb = self.model.extract_embedding_from_pcm(sig.to(torch.float), sr)

        if emb is None or emb.size == 0 or np.isnan(emb).any():
            return None
        return emb.flatten().numpy() if isinstance(emb, torch.Tensor) else emb.flatten()

    def _setup_dirs(self, in_dir: Path, out_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Sets up output directories for each speaker and copies initial files.

        Creates 'definite', 'outliers', and 'uncertain' subdirectories for each speaker.
        Copies all .wav and .txt files from the speaker's input directory to the
        'definite' directory initially.

        Args:
            in_dir: The base input directory (e.g., .../ctc_aligner_output/).
            out_dir: The base output directory for outlier detection results.

        Returns:
            A dictionary mapping speaker names to their relevant paths and audio files,
            or an empty dictionary if no target speaker directories are found.
        """
        data = {}
        
        # Process each target speaker
        for spkr in self.target_speakers:
            # Convert speaker name to directory format
            spkr_dir = spkr.replace(" ", "_")
            src_dir = in_dir / spkr_dir
            spkr_out_dir = out_dir / spkr_dir
            
            if not src_dir.is_dir():
                continue

            filt_dir = spkr_out_dir / self.def_dir
            outlier_dir = spkr_out_dir / self.outlier_dir
            unc_dir = spkr_out_dir / self.unc_dir
            
            filt_dir.mkdir(parents=True, exist_ok=True)
            outlier_dir.mkdir(parents=True, exist_ok=True)
            unc_dir.mkdir(parents=True, exist_ok=True)

            audio_files = []
            files = list(src_dir.glob("*.wav"))
            files.extend(list(src_dir.glob("*.txt")))

            if not files:
                continue
            
            for src in files:
                dst = filt_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                if src.suffix.lower() == ".wav":
                    audio_files.append(dst)

            if audio_files:
                data[spkr] = {
                    'spkr_dir': spkr_out_dir,
                    'filt_dir': filt_dir,
                    'audio_files': sorted(audio_files)
                }
        return data

    def _get_profile(self, files: List[Path], name: str) -> Tuple[Optional[np.ndarray], Dict[Path, np.ndarray]]:
        """Generates a speaker profile (centroid embedding) from a list of audio files.

        Refines the centroid using a percentile-based approach if enough segments are available.

        Args:
            files: A list of Paths to audio files for the speaker.
            name: The name of the speaker (for logging).

        Returns:
            A tuple containing the centroid (NumPy array) and a dictionary mapping
            file paths to their embeddings. Returns (None, embeddings_dict) if a profile
            cannot be generated (e.g., too few files, embedding errors).
        """
        embs = {}
        emb_list = []

        for path in tqdm(files, desc=f"Embedding {name}", leave=False):
            emb = self._get_embedding(path)
            if emb is not None:
                embs[path] = emb
                emb_list.append(emb)

        if len(emb_list) < self.min_profile:
            logger.warning(f"Not enough embeddings for {name}, skipping")
            return None, embs

        if not all(emb.shape == emb_list[0].shape and len(emb.shape) == 1 for emb in emb_list):
            logger.warning(f"Embedding shape mismatch for {name}, skipping")
            return None, embs

        emb_arr = np.array(emb_list)
        centroid = np.mean(emb_arr, axis=0)

        if emb_arr.shape[0] >= self.min_refine:
            logger.info(f"Refining profile for {name}")
            dists = 1.0 - cosine_similarity(centroid.reshape(1, -1), emb_arr)[0]
            if len(dists) > 0:
                thresh = np.percentile(dists, self.refine_pct)
                idx = np.where(dists <= thresh)[0]
                if len(idx) >= self.min_refine:
                    centroid = np.mean(emb_arr[idx], axis=0)
            
        return centroid, embs

    def _process_speaker(
        self, spkr_dir: Path, 
        files: List[Dict[str, Any]], centroid: np.ndarray
    ) -> Tuple[int, int]:
        """Processes audio files for a single speaker against their voice profile centroid.

        Compares each file's embedding to the centroid and moves files to 'outliers'
        or 'uncertain' directories based on configured similarity thresholds.

        Args:
            spkr_dir: The base output directory for the current speaker.
            files: A list of dictionaries, each containing 'path' and 'embedding' for a file.
            centroid: The speaker's voice profile (centroid embedding as a NumPy array).

        Returns:
            A tuple (definite_outliers_moved, uncertain_files_moved).
        """
        
        filt_dir = spkr_dir / self.def_dir
        out_dir = spkr_dir / self.outlier_dir
        unc_dir = spkr_dir / self.unc_dir

        def_moved = unc_moved = 0
        
        unc_thresh = min(self.unc_thresh, self.def_thresh - 0.001)
        centroid_2d = centroid.reshape(1, -1)

        # Vectorized distance computation for all files at once
        all_embs = np.array([item['embedding'] for item in files])
        all_dists = 1.0 - cosine_similarity(centroid_2d, all_embs)[0]

        for item, dist in zip(files, all_dists):
            path = item['path']
            
            if not path.exists():
                continue

            move_dir = None
            is_def = is_unc = False

            if dist > self.def_thresh:
                move_dir, is_def = out_dir, True
            elif dist > unc_thresh:
                move_dir, is_unc = unc_dir, True
            
            if move_dir:
                base = path.name
                txt = path.with_suffix(".txt").name
                txt_src = filt_dir / txt
                
                try:
                    shutil.move(str(path), str(move_dir / base))
                    if txt_src.exists():
                        shutil.move(str(txt_src), str(move_dir / txt))
                    if is_def:
                        def_moved += 1
                    elif is_unc:
                        unc_moved += 1
                except Exception:
                    continue

        return def_moved, unc_moved

    def _print_summary(self, spkr: str, def_moved: int, unc_moved: int, total_files: int):
        """Prints a summary of outlier detection results for a speaker.

        Args:
            spkr: The name of the speaker.
            def_moved: Number of files moved to 'outliers'.
            unc_moved: Number of files moved to 'uncertain'.
            total_files: Total number of files processed for the speaker.
        """
        if total_files == 0:
            return

        def_pct = (def_moved / total_files) * 100
        unc_pct = (unc_moved / total_files) * 100
        
        logger.info(f"\n=== Summary for {spkr} ===")
        logger.info(f"Total files processed: {total_files}")
        logger.info(f"Definite outliers: {def_moved} ({def_pct:.1f}%)")
        logger.info(f"Uncertain files: {unc_moved} ({unc_pct:.1f}%)")

    def process_directory(self, in_dir: Path, out_dir: Path):
        """Processes all target speakers found in the input directory for outlier detection.

        For each speaker, sets up directories, generates a voice profile, processes files
        against the profile, and prints a summary.

        Args:
            in_dir: The base input directory (e.g., .../ctc_aligner_output/)
                    containing subdirectories for each speaker.
            out_dir: The base output directory where outlier detection results will be stored,
                     with subdirectories for each speaker and their categories.
        """
        # Process each speaker
        for spkr in self.target_speakers:
            data = self._setup_dirs(in_dir, out_dir)
            
            if spkr not in data:
                continue

            speaker_data = data[spkr]
            audio_files = speaker_data['audio_files']

            centroid, embs = self._get_profile(audio_files, spkr)
            if centroid is None:
                logger.warning(f"No valid profile found for {spkr}, skipping")
                continue

            files = [{'path': p, 'embedding': embs[p]} for p in audio_files if p in embs]
            if not files:
                logger.warning(f"No valid embeddings found for {spkr}, skipping")
                continue

            def_moved, unc_moved = self._process_speaker(
                speaker_data['spkr_dir'], files, centroid
            )

            self._print_summary(
                spkr, 
                def_moved, unc_moved, len(audio_files)
            )


if __name__ == '__main__':
    with open("configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    base_dir = Path(cfg["project_setup"]["output_base_dir"]) / Path(cfg["project_setup"]["series_name"])
    in_dir = base_dir / Path(cfg["ctc_aligner"]["output_folder"])
    out_dir = base_dir / Path(cfg["outlier_detector"]["output_folder"])
    
    print(f"Testing outlier detection for {in_dir}")
    detector = OutlierDetector(global_config=cfg, outlier_config=cfg["outlier_detector"])
    detector.process_directory(in_dir, out_dir)