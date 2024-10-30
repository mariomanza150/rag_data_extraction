import asyncio
import os
import logging

from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunky import Chunky
from llm_manager import LLModelManager
from utils import (
    get_azure_blob_client,
    get_processed_file_names,
    is_report,
    iter_over_async,
    filter_by_latest,
)

logging.basicConfig(
    filename="embedding_loader.log",
    filemode="w",
    format="%(levelname)s - %(asctime)s - %(message)s",
)


class EmbeddingLoader:
    def __init__(
        self,
        embed_function,
        chroma_path,
        data_path,
        loaded_files_path,
        local_embedding,
        infer_sections,
        local_documents,
        remote_source,
        fetch_remote,
        blob_container,
        schema_name,
        model,
    ):
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
        return set(
            [
                m.get("source")
                for m in Chroma(
                    persist_directory=self.chroma_path,
                    embedding_function=self.embed_func,
                ).get(include=["metadatas"])["metadatas"]
            ]
        )

    def get_all_embedded(self):
        docs = self.set_db()
        return docs

    def check_if_embeddings(self, filename):
        docs = self.set_db()
        if filename in docs:
            return True
        return False

    def sync_load_embeddings(self, **kwargs):
        docs = self.set_db()
        loop = asyncio.get_event_loop()
        async_gen = self.load_embeddings(docs)
        sync_gen = iter_over_async(async_gen, loop)
        return [x for x in sync_gen]

    def gen_load_embeddings(self):
        loop = asyncio.get_event_loop()
        docs = self.set_db()
        async_gen = self.load_embeddings(docs)
        sync_gen = iter_over_async(async_gen, loop)
        return sync_gen

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
                yield {
                    "filename": filename,
                    "message": "",
                }
                continue

            chunks = await self.split_document(llm, filename, document)
            if len(chunks) == 0:
                print(filename, ", Could not extract any text: skipping embed")
                continue
            await self.add_to_chroma(db, filename, chunks)

        llm = None
        llm_manager = None
        print("All Embeddings Loaded!")

    def load_documents(self, processed_docs):
        file_names = []
        if self.local_documents:
            directory = os.fsencode(os.path.join(os.getcwd(), self.data_path))

            file_names = [f.decode("utf-8") for f in os.listdir(directory)]
            print(f"found {len(file_names)} locally, processing...")
        elif self.remote_source == "box":
            fetched_documents = []
            if self.fetch_remote:
                from box_loader import AzureBoxLoader

                fetched_documents = AzureBoxLoader(
                    self.loaded_files_path, self.fetch_remote
                ).load_documents()

            directory = os.fsencode(os.path.join(os.getcwd(), self.loaded_files_path))

            # tries to filter out old revisions and versions of the files, should put a timestamp filter on reception from box
            file_names = filter_by_latest(
                fetched_documents
                + [name.decode("utf-8") for name in os.listdir(directory)]
            )
            print(f"found {len(file_names)} on box storage, processing...")
        elif self.remote_source == "blob":
            storage_client = get_azure_blob_client()
            container = self.blob_container

            container_client = storage_client.get_container_client(container)
            processed_files = get_processed_file_names(self.schema_name)

            for file_name in container_client.list_blob_names():
                if is_report(file_name) and file_name not in processed_files:
                    blob = container_client.get_blob_client(blob=file_name)
                    file_path = self.save_pdf_locally(file_name, blob)
                    yield (
                        file_name,
                        PyPDFLoader(file_path).load(),
                    )

        for f in file_names:
            if f in processed_docs:
                print(f"Found embeddings for {f}, skipping")
                yield (f, None)
            else:
                yield (
                    f,
                    PyPDFLoader(os.path.join(directory.decode("utf-8"), f)).load(),
                )

    def save_pdf_locally(self, file_name, blob_object):
        file_path = f"{self.loaded_files_path}/{file_name}"
        with open(file_path, "wb") as file:
            file.write(blob_object.download_blob().readall())
        return file_path

    async def split_document(self, llm, file, documents: list[Document]):
        """Takes care of chunking the document, invokes Chunky for cleaning text"""
        # chunk size 800, overlap 200 seems to give best results
        txt_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300,
            length_function=len,
            is_separator_regex=False,
        )

        # sending call_llm as false to chunky, will skip adding a section to the chunks
        texts, metadatas = [], []
        chunker = Chunky(llm)
        async for text, metadata in chunker.process_documents(file, documents):
            texts.append(text)
            metadatas.append(metadata)

        return txt_splitter.create_documents(texts, metadatas)

    async def add_to_chroma(self, db, filename, chunks: list[Document]):
        # Calculate Page IDs.
        chunks_with_ids = self.calculate_chunk_ids(filename, chunks)
        # Add or Update the documents.
        existing_items = db.get(
            include=["metadatas"]
        )  # IDs are always included by default
        existing_ids = set(existing_items["ids"])

        # Only add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            # Send to backgound to keep running inference
            print(
                f"Start {len(chunks)} embed for {filename}, current db chunks: {len(existing_ids)}"
            )
            try:
                await db.aadd_documents(new_chunks, ids=new_chunk_ids)
                return {
                    "filename": filename,
                    "message": f"📩 Done embedding {len(new_chunks)} chunks for {filename}",
                }
            except Exception as e:
                return {
                    "filename": filename,
                    "message": f"❌ {filename} Error embedding new chunks: {e}",
                }
        else:
            return {
                "filename": filename,
                "message": f"No new chunks to add {filename}",
            }

    def calculate_chunk_ids(self, filename, chunks):
        # This will create IDs like "data/monopoly.pdf:6:2"
        # Page Source : Page Number : Chunk Index

        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            page = chunk.metadata.get("page")
            current_page_id = f"{filename}:{page}"

            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the chunk ID.
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the page meta-data.
            chunk.metadata["source"] = str(filename)
            chunk.metadata["id"] = str(chunk_id)

        return chunks
