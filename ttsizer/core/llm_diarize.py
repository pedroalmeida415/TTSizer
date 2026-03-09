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

load_dotenv()

logger = get_logger("llm_diarizer")

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
    
    def _process_file(self, norm_path: Path, out_path: Path):
        """Processes a single normalized audio file to generate diarization data.

        Uploads the audio file, sends it to the LLM with the formatted prompt,
        and saves the JSON response. Handles potential JSON decoding errors.

        Args:
            norm_path: Path to the normalized input audio file.
            out_path: Path to save the output JSON diarization data.
        """
        client = genai.Client(api_key=self.api_key)

        logger.info(f"Uploading {norm_path.name}...")
        uploaded = client.files.upload(file=str(norm_path))
        logger.info(f"Uploaded: {uploaded.uri}")

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
        config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            safety_settings=[
                genai_types.SafetySetting(category=c, threshold="BLOCK_NONE")
                for c in [
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                ]
            ],
            response_mime_type="application/json",
            top_p=self.top_p,
        )

        response = client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        json_text = response.candidates[0].content.parts[0].text
        
        try:
            data = json.loads(json_text)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except json.JSONDecodeError:
            debug_path = out_path.with_suffix(".raw_llm_output.txt")
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(json_text)

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
        
        logger.info(f"Processing {len(files)} files in {norm_dir}")

        for norm_path in tqdm(files, desc="Processing files", unit="file"):
            rel_path = norm_path.relative_to(norm_dir)
            out_path = out_dir / rel_path.with_suffix(".json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self._process_file(norm_path, out_path)
        

if __name__ == '__main__':
    with open("configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    base = Path(cfg["project_setup"]["output_base_dir"]) / cfg["project_setup"]["series_name"]
    in_dir = base / cfg["vocals_normalizer"]["output_folder"]
    out_dir = base / cfg["llm_diarizer"]["output_folder"]
    
    print(f"Generating transcriptions for {in_dir}")
    diarizer = LLMDiarizer(global_config=cfg, diarizer_config=cfg["llm_diarizer"])
    diarizer.process_directory(in_dir, out_dir) 