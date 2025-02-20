# rag_pipeline/config/config_manager.py
import json
import os
from typing import Any, Dict

from dotenv import load_dotenv

class ConfigManager:
    """
    Manages configuration loading from environment variables and JSON configuration files.
    """
    def __init__(self) -> None:
        load_dotenv()
        # Load environment variables into self.env
        self.env: Dict[str, str] = dict(os.environ)

        # Load JSON configuration files from the config directory
        self.model_config = self._load_json_config("config/config_model.json")
        # Use config_prompts.json as primary prompts config
        self.prompts_config = self._load_json_config("config/config_prompts.json")
        self.sections_config = self._load_json_config("config/config_sections.json")

        # Default process flags (could be merged with command-line flags)
        self.flags: Dict[str, Any] = {
            "out": "json",
            "reset_embeddings": False,
            "load_embeddings": True,
            "load_expected": False,
            "step_load": True,
            "log_prompts": True,
            "log_expected": False,
            "fine_tune_file": False,
        }

        reporting_schema_name = "box_reporting_rag"

        # Embedding/vector store configuration
        self.config_embed_store: Dict[str, Any] = {
            "local": True,
            "path": "chroma",
            "clear_db": self.flags.get("reset_embeddings", False),
            "uri": "",
            "credentials": {},
        }

        # Loader configuration – using Box as remote source here
        self.config_embed_loader: Dict[str, Any] = {
            "local_embedding": True,
            "chroma_path": "chroma",
            "local_documents": True,
            "remote_source": "box",
            "fetch_remote": 3,
            "blob_container": "report-storage",
            "schema_name": reporting_schema_name,
            "data_path": "data",
            "loaded_files_path": "data",
            "infer_sections": False,
            "model": "nomic-embed-text",
        }

        # Azure/DB configuration
        self.config_azure: Dict[str, Any] = {
            "use_db": False,
            "keep_responses": False,
            "schema_name": reporting_schema_name,
            "box_directory": "",
        }

    def _load_json_config(self, filepath: str) -> Dict[str, Any]:
        """
        Load JSON configuration from the specified file path.
        
        Args:
            filepath (str): Path to the JSON configuration file.
        
        Returns:
            Dict[str, Any]: Parsed JSON configuration as a dictionary.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return {}

    def get_config(self) -> Dict[str, Any]:
        """
        Get consolidated configuration dictionary.
        
        Returns:
            Dict[str, Any]: Consolidated configuration.
        """
        return {
            "env": self.env,
            "flags": self.flags,
            "model_config": self.model_config,
            "prompts_config": self.prompts_config,
            "sections_config": self.sections_config,
            "config_embed_store": self.config_embed_store,
            "config_embed_loader": self.config_embed_loader,
            "config_azure": self.config_azure,
        }
