import asyncio
import json
import os
import subprocess
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)

from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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
    def __init__(self) -> None:
        with open("config_model.json") as f:
            self.config = json.load(f)
        self.max_servers = 2
        self.server_ports = []
        self.active_ports = set()

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def make_openai_call(self, model, query):
        return model.invoke(query)

    def get_datapoint_template(self):
        return QUESTION_TEMPLATE

    def get_datapoint_model(self, parallel=True):
        model_config = self.config.get("question_model")

        if model_config.get("ollama"):
            model_settings = model_config.get("model_settings_ollama")
            model = Ollama(
                model=model_config.get("model"),
                **model_settings.get("sampling"),
                **model_settings.get("generation"),
            )
            return self.serve_parallel_model(model) if parallel else model
        else:
            return ChatOpenAI(
                base_url=model_config.get("base_url"),
                openai_api_key=model_config.get("openai_api_key"),
                model=model_config.get("model"),
                temperature=0.1,
            )

    def get_generic_model(self, model):
        return Ollama(model=model, temperature=0.7)

    def get_embedding_model(self):
        model_config = self.config.get("embeddings_model")
        if model_config.get("ollama"):
            return OllamaEmbeddings(model="nomic-embed-text")
        else:
            return OpenAIEmbeddings(**model_config.get("config", {}))

    # max_model = 6 (vram/ram), threads = 5 (cpu) w/gemma:2b
    def serve_parallel_model(self, model, threads=5, max_models=6):
        url = "127.0.0.1:11435"
        subprocess.Popen(
            ["ollama", "serve"],
            env=dict(os.environ)
            | {
                "OLLAMA_HOST": url,
                "OLLAMA_MAX_LOADED_MODELS": str(max_models),
                "OLLAMA_NUM_PARALLEL": str(threads),
            },
            stdout=open("logfile.log", "w"),
            stderr=open("logerrors.log", "w"),
            shell=True,
        )

        if model:
            setattr(model, "base_url", "http://" + url)
            return model

    def sync_local_batch_job(self, model, prompts: list[PromptTemplate]):
        return asyncio.run(self.local_batch_job(model, prompts))

    async def local_batch_job(self, llm, prompts):
        responses = []
        async for response in llm.abatch_as_completed(prompts):
            responses.append(response)

        return responses

    async def openai_batch_job():
        # Implement
        pass
