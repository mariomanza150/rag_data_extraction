# core/embeddings/vector_store.py
import os
import shutil
from langchain_chroma import Chroma

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
        results = self.db.similarity_search_with_score(
            query,
            k=top_k,
            filter={'source': filters.get('source')},
        )
        section_filter = filters.get("section")
        if section_filter:
            boosted = []
            for doc, score in results:
                if doc.metadata.get("section", "").lower() == section_filter.lower():
                    score -= 0.1  # Boost score if section matches
                boosted.append((doc, score))
            boosted.sort(key=lambda x: x[1])
            results = boosted
        return results

    def delete_file_embed(self, file_name):
        self.db.delete(self.db.get(where={"source": file_name})["ids"])

    def clear_database(self):
        if os.path.exists(self.config["path"]):
            shutil.rmtree(self.config["path"])
