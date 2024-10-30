from dotenv import load_dotenv

from orchestrator import Orchestrator

if __name__ == "__main__":
    args = {
        "out": "csv",  # output type files, does not overwrite
        "reset_embeddings": False,  # Reset the database (deletes embeddings).
        "load_embeddings": True,  # Use to load new embeddings.
        "step_load": True,  # First embed, then infer. Default embeds and streams to infer dps
        "log_prompts": True,  # if true, logs the final prompt used
        "log_expected": False,  # if true, logs the expected response
        "fine_tune_file": False,  # if true, creates a file for fine tuning
    }

    config_embed_store = {
        "local": True,
        "path": "chroma",
        "clear_db": args.get("reset_embeddings", False),
        "uri": "",  # support missing
        "credentials": {},  # used with uri for online vector store
    }

    reporting_schema_name = "box_reporting_rag"

    config_embed_loader = {
        "local_embedding": True,  # set to true for using local embedding model.
        "chroma_path": "chroma",  # path for embeddings db, url suppport missing
        "local_documents": False,  # set to true for using local documents.
        "remote_source": "box",  # set to "box" or "blob" for using either Azure source.
        "fetch_remote": 3,  # set to the number of documents to load from a remote source
        "blob_container": "report-storage",  # if blob set, fetches docs from set container
        "schema_name": reporting_schema_name,  # if blob set, fetches docs from set container
        "data_path": "data",  # path to localfiles pdf folder, url suppport missing
        "loaded_files_path": "data",  # box and blox will save and read docs from this folder
        "infer_sections": False,  # set for inferring titles, WIP.
        "model": "nomic-embed-text",
    }

    config_azure = {
        "keep_responses": True,  # If false, sql table results will be overridden by new llm responses
        "schema_name": reporting_schema_name,
        "box_directory": "",
    }

    load_dotenv()

    orca = Orchestrator(
        args.get("log_prompts", True),
        args.get("log_expected", True),
        args.get("fine_tune_file", False),
        config_embed_store,
        config_embed_loader,
        config_azure,
    )
    orca.load_expected_responses()
    orca.start(
        args.get("load_embeddings", False),
        args.get("step_load", True),
        args.get("out", "csv"),
    )
