import os
import shutil

from langchain_community.vectorstores import Chroma


class VectorStore:
    def __init__(self, config, embed_func) -> None:
        self.config = config

        if config.get("clear_db"):
            print("✨ Clearing Database, Bye Embeddings!")
            self.clear_database()

        self.db = Chroma(
            persist_directory=config["path"],
            embedding_function=embed_func,
        )

    def search_chunks(self, query, filters, top_k=5):
        return self.db.similarity_search_with_score(
            query,
            k=top_k,
            filter=filters,
        )

    def delete_file_embed(self, file_name):
        self.db.delete(self.db.get(where={"source": file_name})["ids"])

    def clear_database(self):
        if os.path.exists(self.config["path"]):
            shutil.rmtree(self.config["path"])
