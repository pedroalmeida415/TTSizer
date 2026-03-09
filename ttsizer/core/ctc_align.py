import warnings
warnings.filterwarnings("ignore")

import json
import soundfile as sf
import re
import math
from collections import defaultdict
from typing import Dict, Optional, Tuple, Any
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import torch
from transformers import AutoModelForCTC, AutoTokenizer
import tempfile
import yaml
from ttsizer.utils.logger import get_logger
from ctc_forced_aligner import (generate_emissions, get_spans,
                               postprocess_results, preprocess_text, load_audio)
from ctc_forced_aligner.alignment_utils import forced_align, merge_repeats

def patched_get_alignments(emissions, tokens, tokenizer):
    """Patched version of get_alignments to fix out-of-bounds target indices and vocab mismatches."""
    assert len(tokens) > 0, "Empty transcript"

    dictionary = tokenizer.get_vocab()
    dictionary = {k.lower(): v for k, v in dictionary.items()}
    dictionary["<star>"] = emissions.size(-1) - 1

    token_indices = [
        dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary and dictionary[c] < emissions.size(-1)
    ]
    if not token_indices:
        logger.warning(f"No valid tokens found for forced alignment: {tokens}")
        return [], [], ""

    blank_id = dictionary.get("<blank>", tokenizer.pad_token_id)

    if not emissions.is_cpu:
        emissions = emissions.cpu()
    targets = np.asarray([token_indices], dtype=np.int64)

    path, scores = forced_align(
        emissions.unsqueeze(0).float().numpy(),
        targets,
        blank=blank_id,
    )
    path = path.squeeze()
    if hasattr(path, "tolist"):
        path = path.tolist()
    if not isinstance(path, list):
        path = [path]

    idx_to_token_map = {v: k for k, v in dictionary.items()}
    segments = merge_repeats(path, idx_to_token_map)
    return segments, scores, idx_to_token_map[blank_id]

logger = get_logger("ctc_aligner")

class CTCAligner:
    """Aligns audio segments with their transcripts using CTC forced alignment.
    
    Uses a pre-trained model to precisely align speech segments with their transcripts,
    handling multiple speakers and supporting both vocal and sound effect segments.
    Outputs aligned audio segments and their corresponding text files.
    """
    def __init__(self, global_config: Dict[str, Any], aligner_config: Dict[str, Any]):
        """Initializes the CTCAligner with global and aligner-specific configurations.

        Args:
            global_config: Dictionary containing global project setup information.
            aligner_config: Dictionary containing configuration specific to the CTC aligner.
        """
        proj = global_config["project_setup"]
        self.target_spkrs = proj["target_speaker_labels"]

        self.model_path = aligner_config["model_name"]
        self.lang = aligner_config["language_code"]
        self.batch_size = aligner_config["batch_size"]
        self.use_gpu = aligner_config["use_gpu"]
        
        self.start_pad = aligner_config["start_pad_seconds"]
        self.end_pad = aligner_config["end_pad_seconds"]
        self.min_words = aligner_config["min_words_per_segment"]
        self.min_duration = aligner_config["min_duration_seconds_segment"]
        
        self.out_fmt = aligner_config.get("output_audio_format", "wav").lower()
        self.out_subtype = aligner_config.get("output_audio_subtype", "PCM_24")

        self.device = torch.device("cuda:0" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Loads the CTC model and tokenizer from the specified path."""
        model_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        self.model = AutoModelForCTC.from_pretrained(self.model_path, torch_dtype=model_dtype).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def _time_to_sec(self, time_str: str) -> float:
        """Converts a time string (e.g., 'MM:SS.ms' or 'SS.ms') to seconds.

        Args:
            time_str: The time string to convert.

        Returns:
            The time in seconds as a float.

        Raises:
            ValueError: If the time string format is invalid.
        """
        parts = time_str.split(':')
        if len(parts) == 3:
            m, s = map(float, parts[1:])
            if s >= 60.0: m += math.floor(s / 60.0); s %= 60.0; 
            return m * 60 + s
        elif len(parts) == 2:
            m, s = map(float, parts)
            if s >= 60: m += math.floor(s/60); s %= 60
            return m * 60 + s
        elif len(parts) == 1: return float(parts[0])
        raise ValueError(f"Invalid time format: {time_str}")

    def _clean_name(self, name: str) -> str:
        """Cleans a string to be used as a valid filename component.

        Removes invalid characters and replaces spaces with underscores.

        Args:
            name: The string to clean.

        Returns:
            The cleaned string.
        """
        return re.sub(r'[\\\\/*?:\"<>|]', '', name).replace(' ', '_')

    def _get_timestamps(self, audio_path: Path, text: str) -> Optional[Tuple[float, float]]:
        """Determines the start and end timestamps of spoken text within an audio file.

        Uses CTC forced alignment to find the precise timing of the provided text.

        Args:
            audio_path: Path to the audio file.
            text: The transcript text to align.

        Returns:
            A tuple (start_time, end_time) in seconds if alignment is successful,
            otherwise None.
        """
        if not text: return None

        wf = load_audio(str(audio_path), dtype=self.model.dtype, device=self.device)
        if wf is None or wf.nelement() == 0:
            return None

        with torch.no_grad():
            emissions, stride = generate_emissions(self.model, wf, batch_size=self.batch_size)
        
        tokens, text_s = preprocess_text(text, romanize=True, language=self.lang, split_size='word', star_frequency='edges')
        segments, scores, blank = patched_get_alignments(emissions, tokens, self.tokenizer)
        spans = get_spans(tokens, segments, blank)
        word_ts = postprocess_results(text_s, spans, stride, scores)

        first = next((seg for seg in word_ts if seg.get('text') and seg['text'] != '<star>'), None)
        last = next((seg for seg in reversed(word_ts) if seg.get('text') and seg['text'] != '<star>'), None)
        
        if first and last and 'start' in first and 'end' in last:
            start, end = first['start'], last['end']
            return (start, end) if end > start else None
        return None

    def _save_segment(self, audio: np.ndarray, text: str, out_path: Path, sr: int):
        """Saves an audio segment and its corresponding transcript to disk.

        Args:
            audio: NumPy array of the audio data.
            text: The transcript text.
            out_path: The base path for the output audio file (text file will have .txt).
            sr: The sample rate of the audio data.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio, sr, subtype=self.out_subtype)
        with open(out_path.with_suffix(".txt"), 'w', encoding='utf-8') as f:
            f.write(text)

    def _process_segment(
        self, 
        segment: Dict[str, Any], 
        audio: np.ndarray, 
        audio_info: Any, 
        out_dir: Path, 
        temp_dir: Path, 
        counters: Dict[str, int], 
        ep_name: str
    ) -> Tuple[bool, bool, bool]: # (is_regular_saved, is_expr_saved, had_align_error)
        """Processes a single transcript segment: aligns, crops, and saves audio.

        Handles regular speech segments by aligning them using _get_timestamps,
        and handles expression/sound-effect segments by using their original timestamps.

        Args:
            segment: Dictionary containing segment details (speaker, start, end, transcript).
            audio: NumPy array of the full episode audio.
            audio_info: Information about the audio file (e.g., samplerate).
            out_dir: The base output directory for processed segments.
            temp_dir: Path to a temporary directory for intermediate files.
            counters: Dictionary to keep track of segment counts per speaker.
            ep_name: The name of the current episode.

        Returns:
            A tuple (is_regular_saved, is_expr_saved, had_align_error) indicating
            if a regular vocal segment was saved, if an expression was saved, and
            if an alignment error occurred for a regular segment.
        """
        spkr = segment.get('speaker')
        start_str = segment.get('start')
        end_str = segment.get('end')
        transcript = segment.get('transcript')

        # Basic validation
        if not (spkr and start_str and end_str): return False, False, False 
        if spkr not in self.target_spkrs: return False, False, False 

        t0 = self._time_to_sec(start_str)
        t1 = self._time_to_sec(end_str)
        if t1 <= t0: return False, False, False
        
        duration = t1 - t0
        is_expr = transcript.startswith(('(', '[')) and transcript.endswith((')', ']'))
        
        start_sec, end_sec = -1.0, -1.0
        align_err = False
        text = transcript if isinstance(transcript, str) else ""

        if is_expr:
            # Expression segments are saved as is
            start_sec, end_sec = t0, t1
        else:
            text = text.strip()
            if not text or len(text.split()) < self.min_words: return False, False, False
            if duration < self.min_duration: return False, False, False
            
            # Initial padded crop for alignment
            t0_pad = max(0.0, t0 - self.start_pad)
            t1_pad = min(audio.shape[0], t1 + self.end_pad)
            if t1_pad <= t0_pad:
                return False, False, False
            
            sr = audio_info.samplerate
            i0, i1 = math.floor(t0_pad * sr), math.ceil(t1_pad * sr)
            if i1 <= i0: return False, False, False

            chunk = audio[i0:i1]
            if chunk.size == 0: return False, False, False

            # Write temp file for alignment
            temp_path = temp_dir / f"temp_{self._clean_name(spkr)}_{counters.get(spkr,0)}_{t0:.3f}.wav"
            sf.write(str(temp_path), chunk, sr)

            # Align
            times = self._get_timestamps(temp_path, text)
            if temp_path.exists(): temp_path.unlink(missing_ok=True) # Clean up temp wav

            if times is None:
                align_err = True
                return False, False, align_err

            rel_start, rel_end = times
            start_sec = t0_pad + rel_start
            end_sec = t0_pad + rel_end

        # Final Cropping and Saving
        sr = audio_info.samplerate
        i0, i1 = math.floor(start_sec * sr), math.ceil(end_sec * sr)
        if i1 <= i0: return False, False, align_err

        final_chunk = audio[i0:i1]
        if final_chunk.size == 0: return False, False, align_err

        spkr_name = self._clean_name(spkr)
        counters[spkr_name] = counters.get(spkr_name, 0) + 1
        count = counters[spkr_name]
        fn = f"{spkr_name}_{self._clean_name(ep_name)}_{count:05d}.{self.out_fmt}"
        
        out_subdir = out_dir / ("expressions" if is_expr else f"{spkr_name}")
        out_path = out_subdir / fn

        self._save_segment(final_chunk, text, out_path, sr)
        return not is_expr, is_expr, align_err

    def _process_episode(
        self, json_path: Path, audio_path: Path, ep_name: str, out_dir: Path
    ) -> Tuple[int, int, int, int]:
        """Processes all segments for a single episode (JSON and audio file pair).

        Loads the audio and transcript, then iterates through segments, calling
        _process_segment for each one.

        Args:
            json_path: Path to the JSON transcript file for the episode.
            audio_path: Path to the corresponding audio file for the episode.
            ep_name: The name of the episode (derived from JSON filename).
            out_dir: The base output directory for processed segments.

        Returns:
            A tuple (vocal_count, sound_count, align_errors, total_segments_in_json).
        """
        
        logger.info(f"\n--- Processing Episode: {ep_name} ---")
        logger.info(f"  JSON: {json_path.name}")
        logger.info(f"  Audio: {audio_path.name}")

        with open(json_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)

        info = sf.info(str(audio_path))
        audio, _ = sf.read(str(audio_path), dtype='float32', always_2d=False)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            counters = defaultdict(int)
            vocal_count = sound_count = 0
            align_errors = 0

            for seg in tqdm(segments, desc="Processing Segments", unit="segment"):
                is_vocal, is_expr, has_error = self._process_segment(
                    seg, audio, info, out_dir, temp_dir, counters, ep_name
                )
                if is_vocal:
                    vocal_count += 1
                if is_expr:
                    sound_count += 1
                if has_error:
                    align_errors += 1

        return vocal_count, sound_count, align_errors, len(segments)

    def process_directory(
        self, json_dir: Path, audio_dir: str, out_dir: Path):
        """Processes all JSON transcript files in a directory for CTC alignment.

        Finds corresponding audio files for each JSON, then processes each episode.
        Aggregates and logs overall statistics.

        Args:
            json_dir: Directory containing JSON transcript files.
            audio_dir: Directory containing corresponding audio files.
            out_dir: Base output directory for all processed segments.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        json_files = sorted(list(json_dir.rglob("*.json")))
        if not json_files:
            logger.warning(f"No JSON files found in {json_dir}")
            return

        logger.info(f"Found {len(json_files)} transcription json files in {json_dir}")

        total_vocals = total_sounds = total_errors = total_segments = 0
        for json_path in tqdm(json_files, desc="Processing Episodes", unit="episode"):
            ep_name = json_path.stem
            rel_path = json_path.relative_to(json_dir)
            
            audio_path = None
            for fmt in ['.flac', '.wav']:
                path = audio_dir / rel_path.with_suffix(fmt)
                if path.exists():
                    audio_path = path
                    break

            if not audio_path:
                logger.warning(f"No audio file found for {ep_name}")
                continue

            vocals, sounds, errors, segments = self._process_episode(
                json_path, audio_path, ep_name, out_dir
            )
            total_vocals += vocals
            total_sounds += sounds
            total_errors += errors
            total_segments += segments

        logger.info(f"\n=== Alignment Summary ===")
        logger.info(f"Total Segments: {total_segments}")
        logger.info(f"Vocal Segments: {total_vocals}")
        logger.info(f"Sound Segments: {total_sounds}")
        logger.info(f"Alignment Errors: {total_errors}")


if __name__ == '__main__':
    with open("configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    base_dir = Path(cfg["project_setup"]["output_base_dir"]) / cfg["project_setup"]["series_name"]
    json_dir = base_dir / cfg["llm_diarizer"]["output_folder"]
    audio_dir = base_dir / cfg["vocals_normalizer"]["output_folder"]
    out_dir = base_dir / cfg["ctc_aligner"]["output_folder"]
    
    print(f"Aligning audio and transcripts for {json_dir}")
    aligner = CTCAligner(global_config=cfg, aligner_config=cfg["ctc_aligner"])
    aligner.process_directory(json_dir, audio_dir, out_dir) 