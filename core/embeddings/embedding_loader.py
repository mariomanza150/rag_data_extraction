# core/embeddings/embedding_loader.py

import asyncio
import os
import logging

from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.processing.chunky import Chunky
from core.llm.llm_manager import LLModelManager
from utils.storage_utils import get_azure_blob_client, filter_by_latest, iter_over_async, chunk

import logging

logger = logging.getLogger(__name__)

class EmbeddingLoader:
    def __init__(self, embed_function, chroma_path, data_path, loaded_files_path, local_embedding, infer_sections, local_documents, remote_source, fetch_remote, blob_container, schema_name, model):
        self.embed_func = embed_function
        self.chroma_path = chroma_path
        self.data_path = data_path
        self.loaded_files_path = loaded_files_path
        self.local_embedding = local_embedding
        self.infer_sections = infer_sections
        self.local_documents = local_documents
        self.remote_source = remote_source
        self.fetch_remote = fetch_remote
        self.blob_container = blob_container
        self.schema_name = schema_name
        self.model = model

    def set_db(self):
        return set([m.get("source") for m in Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embed_func,
        ).get(include=["metadatas"])["metadatas"]])

    def get_all_embedded(self):
        return self.set_db()

    def check_if_embeddings(self, filename):
        return filename in self.set_db()

    def sync_load_embeddings(self, **kwargs):
        docs = self.set_db()
        loop = asyncio.get_event_loop()
        async_gen = self.load_embeddings(docs)
        # Run an async helper to collect all items from the async generator.
        return loop.run_until_complete(self._collect_async_gen(async_gen))

    async def _collect_async_gen(self, async_gen):
        results = []
        async for item in async_gen:
            results.append(item)
        return results

    def gen_load_embeddings(self):
        loop = asyncio.get_event_loop()
        docs = self.set_db()
        async_gen = self.load_embeddings(docs)
        return iter_over_async(async_gen, loop)

    async def load_embeddings(self, processed_docs):
        if self.local_embedding:
            llm = None
            if self.infer_sections:
                llm_manager = LLModelManager()
                llm = llm_manager.get_datapoint_model()
            db = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embed_func,
            )

        for filename, document in self.load_documents(processed_docs):
            if document is None:
                yield {"filename": filename, "message": ""}
                continue
            chunks = await self.split_document(llm, filename, document)
            if len(chunks) == 0:
                logger.warning(filename, ", Could not extract any text: skipping embed")
                continue
            await self.add_to_chroma(db, filename, chunks)
        logger.info("All Embeddings Loaded!")

    def load_documents(self, processed_docs):
        file_names = []
        if self.local_documents:
            from core.loaders.local_loader import LocalPDFLoader
            loader = LocalPDFLoader(self.data_path)
            local_docs = loader.load_documents()
            logger.info(f"Found {len(local_docs)} PDF files locally, processing...")
            for filename, docs in local_docs:
                if filename in processed_docs:
                    logger.info(f"Found embeddings for {filename}, skipping")
                    yield (filename, None)
                else:
                    yield (filename, docs)
            return
        elif self.remote_source == "box":
            fetched_documents = []
            if self.fetch_remote:
                from core.loaders.box_loader import AzureBoxLoader
                fetched_documents = AzureBoxLoader(self.loaded_files_path, self.fetch_remote).load_documents()
            directory = os.path.join(os.getcwd(), self.loaded_files_path)
            file_names = filter_by_latest(fetched_documents + os.listdir(directory))
            logger.info(f"Found {len(file_names)} on box storage, processing...")
        elif self.remote_source == "blob":
            storage_client = get_azure_blob_client()
            container = self.blob_container
            container_client = storage_client.get_container_client(container)
            processed_files = []  # Adjust as needed
            for file_name in container_client.list_blob_names():
                if file_name and file_name not in processed_files:
                    blob = container_client.get_blob_client(blob=file_name)
                    file_path = self.save_pdf_locally(file_name, blob)
                    yield (file_name, PyPDFLoader(file_path).load())
            return

        for f in file_names:
            if f in processed_docs:
                logger.info(f"Found embeddings for {f}, skipping")
                yield (f, None)
            else:
                file_path = os.path.join(os.getcwd(), self.data_path, f)
                yield (f, PyPDFLoader(file_path).load())

    def save_pdf_locally(self, file_name, blob_object):
        file_path = f"{self.loaded_files_path}/{file_name}"
        with open(file_path, "wb") as file:
            file.write(blob_object.download_blob().readall())
        return file_path

    async def split_document(self, llm, file, documents: list[Document]):
        txt_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300,
            length_function=len,
            is_separator_regex=False,
        )
        texts, metadatas = [], []
        chunker = Chunky(llm)
        async for text, metadata in chunker.process_documents(file, documents):
            texts.append(text)
            metadatas.append(metadata)
        return txt_splitter.create_documents(texts, metadatas)

    async def add_to_chroma(self, db, filename, chunks: list[Document]):
        chunks_with_ids = self.calculate_chunk_ids(filename, chunks)
        existing_items = db.get(include=["metadatas"])
        existing_ids = set(existing_items["ids"])
        new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
        if new_chunks:
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            logger.info(f"Start {len(chunks)} embed for {filename}, current db chunks: {len(existing_ids)}")
            try:
                for str_chunks, chunk_ids in chunk(new_chunks, 60, new_chunk_ids):
                    await db.aadd_documents(str_chunks, ids=chunk_ids)
                    logger.info(f"Embeded {len(chunk_ids)}")
                return {"filename": filename, "message": f"Done embedding {len(new_chunks)} chunks for {filename}"}
            except Exception as e:
                return {"filename": filename, "message": f"Error embedding new chunks: {e}"}
        else:
            return {"filename": filename, "message": f"No new chunks to add {filename}"}

    def calculate_chunk_ids(self, filename, chunks):
        last_page_id = None
        current_chunk_index = 0
        for chunk in chunks:
            page = chunk.metadata.get("page")
            current_page_id = f"{filename}:{page}"
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id
            chunk.metadata["source"] = str(filename)
            chunk.metadata["id"] = str(chunk_id)
        return chunks
