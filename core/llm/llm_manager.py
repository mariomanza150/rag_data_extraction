# rag_pipeline/core/llm/llm_manager.py
import asyncio
import json
import os
import subprocess
import logging
from typing import Any, List

from tenacity import retry, stop_after_attempt, wait_random_exponential

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = logging.getLogger(__name__)

QUESTION_TEMPLATE = """
# CONTEXT #
From a chunk of text, extract the data that best matches the DATAPOINT.
Below are the details of the DATAPOINT you need to extract:
{question}
########
# OBJECTIVE #
Extract the DATAPOINT from the following text:
```{context}```
########
# STYLE #
Keep your answer succinct and cohesive.
Only if the text does not contain the DATAPOINT explicitly: then extrapolate it,
summarize and respond with simple, compound, and compound-complex sentences,
that best answer the QUESTION you are looking for.
To help you extrapolate, you can use the datapoint's DESCRIPTION.

If the DATAPOINT is not present in the text or you cannot extrapolate it, answer with "Not found".
########
# RESPONSE #
Respond only in a valid json format.
The json will contain a key "extracted" and its value is your answer.
"""

class LLModelManager:
    """
    Manages interactions with language models for datapoint extraction and embedding.
    """
    def __init__(self) -> None:
        with open("config/config_model.json", encoding="utf-8") as f:
            self.config = json.load(f)
        self.max_servers = 2
        self.server_ports = []
        self.active_ports = set()

        try:
            # If there's already a running loop (e.g., in Jupyter), get it.
            self.loop = asyncio.get_running_loop()
            logger.info("Using the existing running event loop.")
        except RuntimeError:
            # Otherwise, create a new event loop if one does not exist.
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            logger.info("Created a new event loop.")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def make_openai_call(self, model: Any, query: str) -> Any:
        """
        Make a call to the OpenAI model.
        
        Args:
            model (Any): The language model instance.
            query (str): The prompt query.
        
        Returns:
            Any: The model's response.
        """
        return model.invoke(query)

    def get_datapoint_template(self) -> str:
        """
        Get the template for datapoint extraction.
        
        Returns:
            str: The datapoint extraction template.
        """
        return QUESTION_TEMPLATE

    def get_datapoint_model(self, parallel: bool = True) -> Any:
        """
        Get the language model for datapoint extraction.
        
        Args:
            parallel (bool, optional): Whether to serve parallel model. Defaults to True.
        
        Returns:
            Any: The language model instance.
        """
        model_config = self.config.get("question_model")
        if model_config.get("ollama"):
            model_settings = model_config.get("model_settings_ollama")
            model = OllamaLLM(
                model=model_config.get("model"),
                **model_settings.get("sampling"),
                **model_settings.get("generation"),
            )
            return self.serve_parallel_model(model) if parallel else model
        else:
            return
            return ChatOpenAI(
                base_url=model_config.get("base_url"),
                openai_api_key=model_config.get("openai_api_key"),
                model=model_config.get("model"),
                temperature=0.1,
            )

    def get_generic_model(self, model: str) -> Any:
        """
        Get a generic model instance.
        
        Args:
            model (str): Model name.
        
        Returns:
            Any: The generic model instance.
        """
        return OllamaLLM(model=model, temperature=0.7)

    def get_embedding_model(self) -> Any:
        """
        Get the embedding model based on configuration.
        
        Returns:
            Any: The embedding model instance.
        """
        model_config = self.config.get("embeddings_model")
        if model_config.get("ollama"):
            return OllamaEmbeddings(model="nomic-embed-text")
        else:
            return # OpenAIEmbeddings(**model_config.get("config", {}))

    def serve_parallel_model(self, model: Any, threads: int = 5, max_models: int = 6) -> Any:
        """
        Serve a parallel model instance.
        
        Args:
            model (Any): The language model instance.
            threads (int, optional): Number of parallel threads. Defaults to 5.
            max_models (int, optional): Maximum models loaded. Defaults to 6.
        
        Returns:
            Any: The served parallel model instance.
        """
        url = "127.0.0.1:11435"
        subprocess.Popen(
            ["ollama", "serve"],
            env={**os.environ, "OLLAMA_HOST": url, "OLLAMA_MAX_LOADED_MODELS": str(max_models), "OLLAMA_NUM_PARALLEL": str(threads)},
            stdout=open("logfile.log", "w"),
            stderr=open("logerrors.log", "w"),
            shell=True,
        )
        if model:
            setattr(model, "base_url", "http://" + url)
            return model

    def sync_local_batch_job(self, model: Any, prompts: List[PromptTemplate]) -> List[Any]:
        """
        Run local batch job for LLM inference synchronously.
        
        Args:
            model (Any): The language model.
            prompts (List[PromptTemplate]): List of prompt templates.
        
        Returns:
            List[Any]: List of responses.
        """
        return self.loop.run_until_complete(self.local_batch_job(model, prompts))

    async def local_batch_job(self, llm: Any, prompts: List[PromptTemplate]) -> List[Any]:
        """
        Asynchronously run batch job for LLM inference.
        
        Args:
            llm (Any): The language model.
            prompts (List[PromptTemplate]): List of prompt templates.
        
        Returns:
            List[Any]: List of responses.
        """
        responses = []
        async for response in llm.abatch_as_completed(prompts):
            responses.append(response)
        return responses

    async def openai_batch_job(self) -> None:
        """
        Placeholder for OpenAI batch job.
        """
        pass
