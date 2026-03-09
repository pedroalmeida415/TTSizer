import warnings
warnings.filterwarnings("ignore")

import os
import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm.auto import tqdm
from dotenv import load_dotenv
import yaml
from google import genai
from google.genai import types as genai_types
from ttsizer.utils.logger import get_logger
import logging
from dotenv import load_dotenv

load_dotenv()

# Suppress noisy Third-Party module logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("google_genai._api_client").setLevel(logging.WARNING)

logger = get_logger("llm_diarizer")

from pydantic import BaseModel, Field, TypeAdapter
from typing import Optional, List

class DiarizationSegment(BaseModel):
    start: str = Field(description="The precise start time in MM:SS.mmm format")
    end: str = Field(description="The precise end time in MM:SS.mmm format")
    speaker: str = Field(description="The single, consistent label determined for the speaker, or SOUND for non-speech/SFX/silence.")
    transcript: Optional[str] = Field(description="Verbatim text with punctuation context. Use null if SOUND.")

SegmentListAdapter = TypeAdapter(List[DiarizationSegment])

class LLMDiarizer:
    """Handles diarization of audio files using a large language model (LLM).
    
    Uses a specified Gemini model to analyze audio and generate speaker diarization data.
    It formats prompts based on a template and project-specific details, processes audio
    files by uploading them to the LLM, and saves the resulting JSON output.
    """
    def __init__(self, global_config: Dict[str, Any], diarizer_config: Dict[str, Any]):
        """Initializes the LLMDiarizer with global and diarizer-specific configurations.

        Args:
            global_config: Dictionary containing global project setup information.
            diarizer_config: Dictionary containing configuration specific to the diarizer.

        Raises:
            FileNotFoundError: If the prompt template file specified in config is not found.
        """
        self.proj = global_config["project_setup"]

        self.model_name = diarizer_config["model_name"]
        self.temperature = diarizer_config["temperature"]
        self.top_p = diarizer_config["top_p"]
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.skip = diarizer_config.get("skip_if_output_exists", True)
        self.max_workers = diarizer_config.get("max_workers", 1)

        self.tmpl_file = Path(diarizer_config["prompt_template_file"])
        if not self.tmpl_file.exists():
            raise FileNotFoundError(f"Template not found: {self.tmpl_file}")
        self.prompt = self._format_prompt(self.proj["series_name"], self.proj["target_speaker_labels"])
    
    def _format_prompt(self, series_name: str, target_speaker_labels: List[str]) -> str:
        """Formats the prompt string using a template and provided speaker labels.

        Args:
            series_name: The name of the series or project.
            target_speaker_labels: A list of target speaker labels for diarization.

        Returns:
            The formatted prompt string.
        """
        template_text = self.tmpl_file.read_text(encoding='utf-8')
        template_chars = "[" + ", ".join(target_speaker_labels) + "]"
        prompt = (
            template_text
            .replace("{SERIES_NAME}", series_name)
            .replace("{TARGET_SPEAKER_LABELS}", template_chars)
            .replace("{CHARACTER_1}", target_speaker_labels[0])
        )
        logger.info(prompt)
        return prompt
    
    def _process_file(self, norm_path: Path, out_path: Path, worker_id: int):
        """Processes a single normalized audio file to generate diarization data.

        Uploads the audio file, sends it to the LLM with the formatted prompt,
        and saves the JSON response. Handles potential JSON decoding errors.

        Args:
            norm_path: Path to the normalized input audio file.
            out_path: Path to save the output JSON diarization data.
            worker_id: The ID of the threaded worker for UI positioning
        """
        client = genai.Client(api_key=self.api_key)

        pbar = tqdm(total=0, bar_format="{desc}", position=worker_id, leave=False)
        
        import threading
        import time
        
        is_waiting = True
        def animate_waiting():
            dots = 0
            while is_waiting:
                pbar.set_description(f"Wait: {norm_path.name[:18]}...{('.' * dots).ljust(3)}")
                time.sleep(0.5)
                dots = (dots + 1) % 4
                
        anim_thread = threading.Thread(target=animate_waiting)
        anim_thread.start()

        uploaded = client.files.upload(file=str(norm_path), config={'mime_type': 'audio/flac'})

        contents = [
            genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part.from_uri(
                        file_uri=uploaded.uri,
                        mime_type=uploaded.mime_type,
                    ),
                    genai_types.Part.from_text(text=self.prompt),
                ],
            )
        ]
        config = {
            "temperature": self.temperature,
            "safety_settings": [
                {"category": c, "threshold": "BLOCK_NONE"}
                for c in [
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                ]
            ],
            "response_mime_type": "application/json",
            "response_json_schema": SegmentListAdapter.json_schema(),
            "top_p": self.top_p,
        }

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            
            try:
                json_text = response.text or ""
            except ValueError:
                # `response.text` raises ValueError if blocked or empty
                json_text = ""
                logger.warning(f"Response blocked or empty for {norm_path.name}. Prompt Feedback: {getattr(response, 'prompt_feedback', 'No feedback.')}")
            
            try:
                segments = SegmentListAdapter.validate_json(json_text)
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump([s.model_dump() for s in segments], f, indent=4, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Failed to parse structured output for {norm_path.name}: {e}")
                debug_path = out_path.with_suffix(".raw_llm_output.txt")
                with open(debug_path, 'w', encoding='utf-8') as f:
                    f.write(json_text)
        finally:
            is_waiting = False
            anim_thread.join()
            pbar.close()
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass

    def process_directory(self, norm_dir: Path, out_dir: Path):
        """Processes all audio files (FLAC or WAV) in a given directory for diarization.

        Iterates through audio files, calls _process_file for each, and saves outputs.

        Args:
            norm_dir: Path to the directory containing normalized audio files.
            out_dir: Path to the directory where output JSON files will be saved.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(list(norm_dir.rglob("*.flac")))
        if not files:
            files = sorted(list(norm_dir.rglob("*.wav")))
        
        if not files:
            logger.warning(f"No files found in {norm_dir}")
            return
        
        logger.info(f"Processing {len(files)} files in {norm_dir} with {self.max_workers} workers")

        import concurrent.futures
        import queue

        worker_queue = queue.Queue()
        for i in range(1, self.max_workers + 1):
            worker_queue.put(i)

        def process_wrapper(norm_path: Path):
            rel_path = norm_path.relative_to(norm_dir)
            out_path = out_dir / rel_path.with_suffix(".json")
            if self.skip and out_path.exists():
                return
            out_path.parent.mkdir(parents=True, exist_ok=True)
            worker_id = worker_queue.get()
            try:
                self._process_file(norm_path, out_path, worker_id)
            finally:
                worker_queue.put(worker_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {executor.submit(process_wrapper, p): p for p in files}
            
            # Use tqdm with as_completed to track progress of concurrent tasks
            for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(files), desc="Processing files", unit="file", position=0):
                try:
                    future.result()
                except Exception as exc:
                    path = future_to_path[future]
                    logger.error(f"File {path.name} generated an exception: {exc}", exc_info=True)
        

if __name__ == '__main__':
    with open("configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    base = Path(cfg["project_setup"]["output_base_dir"]) / cfg["project_setup"]["series_name"]
    in_dir = base / cfg["vocals_normalizer"]["output_folder"]
    out_dir = base / cfg["llm_diarizer"]["output_folder"]
    
    print(f"Generating transcriptions for {in_dir}")
    diarizer = LLMDiarizer(global_config=cfg, diarizer_config=cfg["llm_diarizer"])
    diarizer.process_directory(in_dir, out_dir) 