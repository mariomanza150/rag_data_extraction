# main.py
import logging
from config.config_manager import ConfigManager
from core.orchestrator import Orchestrator

def main() -> None:
    """Main entry point for the RAG pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s" # %(asctime)s - 
    )
    config_manager = ConfigManager()
    config = config_manager.get_config()

    flags = config["flags"]

    orchestrator = Orchestrator(
        log_prompts=flags.get("log_prompts", True),
        log_expected=flags.get("log_expected", False),
        fine_tune_file=flags.get("fine_tune_file", False),
        es_config=config["config_embed_store"],
        el_config=config["config_embed_loader"],
        az_config=config["config_azure"],
    )
    if flags.get("load_expected"):
        orchestrator.load_expected_responses()
    orchestrator.start(
        load_embed=flags.get("load_embeddings", True),
        step_load=flags.get("step_load", True),
        out=flags.get("out", "csv"),
    )

if __name__ == "__main__":
    main()
