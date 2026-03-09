#!/usr/bin/env python3
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ttsizer.core.audio_extract import AudioExtractor
from ttsizer.core.vocals_extract import VocalsExtractor
from ttsizer.core.vocals_normalize import VocalsNormalizer
from ttsizer.core.llm_diarize import LLMDiarizer
from ttsizer.core.ctc_align import CTCAligner
from ttsizer.core.outlier_detect import OutlierDetector
from ttsizer.core.asr_process import ASRProcessor
from ttsizer.utils.logger import get_logger, initialize_logging

logger: Optional[logging.Logger] = None

# Define the order of pipeline stages
STAGES = [
    "audio_extractor",
    "vocals_extractor",
    "vocals_normalizer",
    "llm_diarizer",
    "ctc_aligner",
    "outlier_detector",
    "asr_processor"
]

class PipelineOrchestrator:
    """Manages and executes the TTSizer processing pipeline stages."""
    def __init__(self, cfg_path_str: str):
        """Initializes the orchestrator with pipeline and project configurations.

        Args:
            cfg_path_str: Path to the main YAML configuration file.
        """
        self.cfg_path = Path(cfg_path_str).resolve()
        self.root = self.cfg_path.parent.parent

        with open(self.cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        self.setup = self.cfg.get("project_setup")
        if not self.setup or not isinstance(self.setup, dict):
            raise ValueError("'project_setup' section is missing or invalid in the config file.")
            
        self.series_name = self.setup.get("series_name")
        if not self.series_name:
            raise ValueError("'series_name' is missing in 'project_setup'.")

        self.video_input_base_dir = Path(self.setup["video_input_base_dir"])
        self.output_base_dir = Path(self.setup["output_base_dir"])

        if not self.video_input_base_dir.is_absolute():
            self.video_input_base_dir = (self.root / self.video_input_base_dir).resolve()
        if not self.output_base_dir.is_absolute():
            self.output_base_dir = (self.root / self.output_base_dir).resolve()
            
        self.project_output_dir = (self.output_base_dir / self.series_name).resolve()

        if Path.cwd() != self.root:
            logger.info(f"Changing working directory from {Path.cwd()} to {self.root}")
            os.chdir(self.root)
        
        self.runners = {
            STAGES[0]: self._run_audio_extract,
            STAGES[1]: self._run_vocals_extract,
            STAGES[2]: self._run_vocals_normalize,
            STAGES[3]: self._run_llm_diarize,
            STAGES[4]: self._run_ctc_align,
            STAGES[5]: self._run_outlier_detect,
            STAGES[6]: self._run_asr_process,
        }
        
        self.pipeline_control = self.cfg.get("pipeline_control", {})

    def _get_stage_cfg(self, stage_key: str) -> Dict[str, Any]:
        """Retrieves the configuration dictionary for a specific pipeline stage.

        Args:
            stage_key: The key identifying the pipeline stage (e.g., 'audio_extractor').

        Returns:
            Configuration dictionary for the stage, or an empty dict if not found.
        """
        cfg = self.cfg.get(stage_key)
        if not cfg or not isinstance(cfg, dict):
            # Log a warning but proceed, stage run methods should handle missing config gracefully or error out.
            logger.warning(f"Configuration for stage '{stage_key}' is missing or invalid in the config file.")
            return {}
        return cfg

    def _get_stage_io_paths(self, current_stage_key: str, prev_stage_key: Optional[str] = None) -> Tuple[Optional[Path], Path]:
        """Determines input and output paths for a pipeline stage.

        Args:
            current_stage_key: Key of the current stage.
            prev_stage_key: Key of the preceding stage, if any.

        Returns:
            A tuple containing the input Path (or None) and the output Path.
        
        Raises:
            ValueError: If 'output_folder' is missing in the stage configuration.
        """
        current_cfg = self._get_stage_cfg(current_stage_key)
        
        current_output_folder = current_cfg.get("output_folder")
        if not current_output_folder:
            # This is a critical error for a stage to function, so raise it.
            msg = f"Configuration for '{current_stage_key}' is missing 'output_folder'."
            logger.error(msg)
            raise ValueError(msg)
        output_path = self.project_output_dir / current_output_folder

        input_path = None
        if prev_stage_key:
            prev_cfg = self._get_stage_cfg(prev_stage_key)
            prev_output_folder = prev_cfg.get("output_folder")
            if not prev_output_folder:
                msg = f"Configuration for previous stage '{prev_stage_key}' (needed by '{current_stage_key}') is missing 'output_folder'."
                logger.error(msg)
                raise ValueError(msg)
            input_path = self.project_output_dir / prev_output_folder
        
        return input_path, output_path

    def _run_audio_extract(self):
        """Executes the audio extraction stage of the pipeline."""
        stage_key = STAGES[0]
        logger.info(f"\n--- Running Stage: {stage_key} ---")
        config = self._get_stage_cfg(stage_key)
        if not config: logger.warning(f"Skipping stage {stage_key} due to missing config."); return

        video_series_dir = self.video_input_base_dir / self.series_name
        _, output_path = self._get_stage_io_paths(stage_key)

        extractor = AudioExtractor(config)
        extractor.process_directory(video_series_dir, output_path)
        logger.info(f"--- Stage {stage_key} complete ---")

    def _run_vocals_extract(self):
        """Executes the vocal extraction stage of the pipeline."""
        stage_key = STAGES[1]
        logger.info(f"\n--- Running Stage: {stage_key} ---")
        config = self._get_stage_cfg(stage_key)
        if not config: logger.warning(f"Skipping stage {stage_key} due to missing config."); return
        
        input_path, output_path = self._get_stage_io_paths(stage_key, STAGES[0])
        if not input_path: logger.error(f"Cannot run {stage_key}, input path from {STAGES[0]} is missing."); return

        extractor = VocalsExtractor(config)
        extractor.process_directory(input_path, output_path)
        logger.info(f"--- Stage {stage_key} complete ---")

    def _run_vocals_normalize(self):
        """Executes the vocal normalization stage of the pipeline."""
        stage_key = STAGES[2]
        logger.info(f"\n--- Running Stage: {stage_key} ---")
        config = self._get_stage_cfg(stage_key)
        if not config: logger.warning(f"Skipping stage {stage_key} due to missing config."); return
        
        input_path, output_path = self._get_stage_io_paths(stage_key, STAGES[1])
        if not input_path: logger.error(f"Cannot run {stage_key}, input path from {STAGES[1]} is missing."); return
        
        normalizer = VocalsNormalizer(config)
        normalizer.process_directory(input_path, output_path)
        logger.info(f"--- Stage {stage_key} complete ---")

    def _run_llm_diarize(self):
        """Executes the LLM-based diarization and transcription stage."""
        stage_key = STAGES[3]
        logger.info(f"\n--- Running Stage: {stage_key} ---")
        config = self._get_stage_cfg(stage_key)
        if not config: logger.warning(f"Skipping stage {stage_key} due to missing config."); return
        
        input_path, output_path = self._get_stage_io_paths(stage_key, STAGES[2])
        if not input_path: logger.error(f"Cannot run {stage_key}, input path from {STAGES[2]} is missing."); return

        diarizer = LLMDiarizer(self.cfg, config)
        diarizer.process_directory(input_path, output_path)
        logger.info(f"--- Stage {stage_key} complete ---")

    def _run_ctc_align(self):
        """Executes the CTC forced alignment stage."""
        stage_key = STAGES[4]
        logger.info(f"\n--- Running Stage: {stage_key} ---")
        config = self._get_stage_cfg(stage_key)
        if not config: logger.warning(f"Skipping stage {stage_key} due to missing config."); return

        transcriptions_input_path, output_path = self._get_stage_io_paths(stage_key, STAGES[3])
        if not transcriptions_input_path: logger.error(f"Cannot run {stage_key}, input path from {STAGES[3]} is missing."); return

        vocals_norm_cfg = self._get_stage_cfg(STAGES[2])
        vocals_norm_output_folder = vocals_norm_cfg.get("output_folder")
        if not vocals_norm_output_folder: 
             msg = f"Missing output_folder for audio source stage '{STAGES[2]}' needed by '{stage_key}'."
             logger.error(msg)
             raise ValueError(msg)
        audio_input_path = self.project_output_dir / vocals_norm_output_folder
        
        aligner = CTCAligner(self.cfg, config)
        aligner.process_directory(transcriptions_input_path, audio_input_path, output_path)
        logger.info(f"--- Stage {stage_key} complete ---")

    def _run_outlier_detect(self):
        """Executes the outlier detection stage for speaker voice profiles."""
        stage_key = STAGES[5]
        logger.info(f"\n--- Running Stage: {stage_key} ---")
        config = self._get_stage_cfg(stage_key)
        if not config: logger.warning(f"Skipping stage {stage_key} due to missing config."); return

        input_path, output_path = self._get_stage_io_paths(stage_key, STAGES[4])
        if not input_path: logger.error(f"Cannot run {stage_key}, input path from {STAGES[4]} is missing."); return

        detector = OutlierDetector(self.cfg, config)
        detector.process_directory(input_path, output_path)
        logger.info(f"--- Stage {stage_key} complete ---")

    def _run_asr_process(self):
        """Executes the ASR processing stage for final transcription and flagging."""
        stage_key = STAGES[6]
        logger.info(f"\n--- Running Stage: {stage_key} ---")
        config = self._get_stage_cfg(stage_key)
        if not config: logger.warning(f"Skipping stage {stage_key} due to missing config."); return

        input_path, output_path = self._get_stage_io_paths(stage_key, STAGES[5])
        if not input_path: logger.error(f"Cannot run {stage_key}, input path from {STAGES[5]} is missing."); return

        processor = ASRProcessor(self.cfg, config)
        processor.process_directory(input_path, output_path) 
        logger.info(f"--- Stage {stage_key} complete ---")

    def run(self):
        """Runs the configured pipeline stages based on pipeline_control settings."""
        run_only = self.pipeline_control.get("run_only_stage")
        start_stage = self.pipeline_control.get("start_stage")
        end_stage = self.pipeline_control.get("end_stage")

        active_stages = []

        if run_only:
            if run_only not in STAGES:
                logger.error(f"Stage '{run_only}' in 'run_only_stage' is not a valid stage name.")
                return
            if run_only not in self.runners:
                logger.error(f"Runner for stage '{run_only}' not implemented.")
                return
            active_stages = [run_only]
        else:
            start_index = STAGES.index(start_stage) if start_stage and start_stage in STAGES else 0
            end_index = STAGES.index(end_stage) if end_stage and end_stage in STAGES else len(STAGES) - 1
            
            if start_index > end_index:
                logger.error("start_stage cannot be after end_stage.")
                return
            active_stages = STAGES[start_index : end_index + 1]

        logger.info(f"Executing stages: {active_stages}")

        for stage_key in active_stages:
            if stage_key in self.runners:
                try:
                    self.runners[stage_key]()
                except Exception as e:
                    logger.error(f"Error during execution of stage '{stage_key}': {e}", exc_info=True)
                    logger.error(f"Pipeline execution halted due to error in stage '{stage_key}'.")
                    return
            else:
                logger.warning(f"Runner for stage '{stage_key}' not implemented or stage not recognized.")
        
        logger.info("\n✨ TTSizer pipeline finished! ✨")

def main():
    """Main entry point for the TTSizer pipeline execution."""
    global logger
    cfg_path_str = "configs/config.yaml" 
    
    initialize_logging(cfg_path_str)
    logger = get_logger("orchestrator") 

    if not Path(cfg_path_str).exists():
        logger.critical(f"Configuration file {Path(cfg_path_str).resolve()} was not found. Exiting.")
        sys.exit(1)
        
    try:
        logger.info(f"Starting TTSizer pipeline with config: {Path(cfg_path_str).resolve()}")
        orchestrator = PipelineOrchestrator(cfg_path_str)
        orchestrator.run()
    except ValueError as ve: 
        logger.critical(f"Pipeline halted due to a configuration or critical setup error: {ve}", exc_info=True)
        sys.exit(1)
    except FileNotFoundError as fnfe:
        logger.critical(f"Pipeline halted because a required file was not found: {fnfe}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"☠️ An unexpected fatal error occurred in the pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
