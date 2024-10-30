import asyncio
import copy
import csv
import json
import os
import time
import traceback

from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from embedding_loader import EmbeddingLoader
from llm_manager import LLModelManager
from utils import (
    get_lat_lng,
    load_extracted_points_to_sql_server,
    create_schema_tables,
    get_processed_file_names,
)
from vector_store import VectorStore


class Orchestrator:
    """Class to orchestrate models, vectorstores and embeddings
    Also handles prompt generation and results storing.
    """

    def __init__(
        self,
        log_prompts: bool,
        log_expected: bool,
        fine_tune_file: bool,
        es_config: dict,
        el_config: dict,
        az_config: dict,
    ):
        self.llm_manager = LLModelManager()

        self.datapoint_template = self.llm_manager.get_datapoint_template()
        self.datapoint_llm = self.llm_manager.get_datapoint_model()
        self.datapoint_prompts = []

        self.es_config = es_config
        self.vector_store = VectorStore(
            es_config, self.llm_manager.get_embedding_model()
        )

        self.embedder = EmbeddingLoader(
            self.llm_manager.get_embedding_model(), **el_config
        )

        self.azure_config = az_config

        self.log_prompt = log_prompts
        self.log_expected = log_expected
        self.overwrite = True
        self.fine_tune_file = fine_tune_file

    def dp_prompt(self, db_results, query):
        """Joins embedding search results with datapoint query
        Uses the template defined in QUESTION_TEMPLATE in llm_manager.py
        """
        if isinstance(self.datapoint_llm, Ollama):
            template = ChatPromptTemplate.from_template(self.datapoint_template)
        elif isinstance(self.datapoint_llm, ChatOpenAI):
            template = PromptTemplate.from_template(self.datapoint_template)
        context_text = "\n".join([doc.page_content for doc, _score in db_results])
        return template.format(context=context_text, question=query)

    async def create_prompt(self, file_name, datapoint):
        """Extracts the relevant embeddings and returns the final prompt."""
        try:
            filters = {"source": file_name}
            # When adding sections to the embedding metadata, this search will be better
            # filters = {
            #    "$and": [{"source": file_name}, {"section": datapoint["section"]}]
            # }
            conf = copy.deepcopy(self.es_config)
            conf.pop("clear_db")
            vs = VectorStore(conf, self.llm_manager.get_embedding_model())

            results = []
            for search in datapoint["search"].split("/"):
                keywords = " ".join(search.split(","))
                # top_k defines the number of embeddings returned by the search
                results += vs.search_chunks(keywords, filters, top_k=3)

            if results:
                query = datapoint["question"]
                return {
                    "dp": datapoint,
                    "prompt": self.dp_prompt(results, query),
                }
            else:
                return {
                    "dp": {"datapoint": datapoint["slug_name"]},
                    "response": "no related chunks found",
                }
        except RuntimeError:
            print(f"error with embeddings, deleting for {file_name}")
            self.vector_store.delete_file_embed(file_name)

    async def process_datapoints(self, file_name, datapoints_prompts):
        """Async function to speed up final prompt generation."""
        prompts = []

        async with asyncio.TaskGroup() as tg:
            for datapoint in datapoints_prompts:
                prompt = tg.create_task(
                    self.create_prompt(file_name, datapoint),
                )
                # prompt.add_done_callback(lambda t: print(t.result()))
                prompts.append(prompt)
                await asyncio.sleep(0)

        return await asyncio.gather(*prompts, return_exceptions=True)

    def run_dp_inference(self, fold_out, file_name, out):
        """Runs inference for prompts and writes results to file.
        fold_out: str, path to output folder
        file_name: str, filename of response file
        out: str, output file type (json or csv)
        """
        clean_filename = file_name.replace(".pdf", "").replace("-Signed", "").strip()
        resp_filename = f"{fold_out}{clean_filename}.{out}"
        if os.path.exists(resp_filename) and not self.overwrite:
            print(f"Skipping, {clean_filename} response file found")
            return

        print(f"Searching for DP's from: {clean_filename}")
        print("formatting prompts...")
        dpoints = asyncio.run(
            self.process_datapoints(file_name, self.datapoint_prompts)
        )

        not_found, found_data, prompts = [], [], []
        for dp in dpoints:
            if "prompt" in dp:
                found_data.append(dp)
                prompts.append(dp["prompt"])
            else:
                not_found.append(dp)

        print(
            "formatting prompts complete",
            len(found_data),
            "of",
            len(self.datapoint_prompts),
        )

        print("getting responses...")
        start = time.time()
        raw_responses = []
        if isinstance(self.datapoint_llm, Ollama):
            raw_responses = self.llm_manager.sync_local_batch_job(
                self.datapoint_llm, prompts
            )
        elif isinstance(self.datapoint_llm, ChatOpenAI):
            for idx, p in enumerate(prompts):
                raw_responses.append(
                    (idx, self.llm_manager.make_openai_call(self.datapoint_llm, p))
                )
            # raw_responses = [resp for resp in self.datapoint_llm.batch_as_completed(prompts)]

        print(
            f"⏲️ time taken for {len(raw_responses)}  responses: {time.time() - start}"
        )

        responses, dict_resp = [], []
        for resp in raw_responses:
            dp = found_data[resp[0]]
            name = dp["dp"].get("slug_name")

            prompt = ""
            if self.log_prompt:
                prompt = dp.get("prompt")
            expected = ""
            if self.log_expected:
                expected = self.get_expected(clean_filename, name)

            resp = resp[1] if isinstance(resp[1], str) else resp[1].content
            try:
                json_resp = json.loads(resp)
                response = json_resp.get("extracted", json_resp.get(name, json_resp))
            except json.decoder.JSONDecodeError:
                response = resp

            if name == "site_address":
                try:
                    lat_lng = get_lat_lng(response)
                    if lat_lng:
                        response = lat_lng
                except Exception as e:
                    print(f"❌ error in {name} get_lat_lng: {e}")

            responses.append([name, response, expected, prompt])
            dict_resp.append(
                {
                    "name": name,
                    "response": response,
                    "expected": expected,
                    "prompt": prompt,
                }
            )

        resp_filename = f"all_responses.{out}"
        file_mode = "a"
        print("writing responses...")
        if out == "csv":
            print(resp_filename)
            with open(resp_filename, file_mode, newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["datapoint", "actual_response", "expected_response"])
                writer.writerows(responses)
        elif out == "json":
            print(resp_filename)
            with open(resp_filename, file_mode, encoding="utf-8") as f:
                json.dump(dict_resp, f, indent=2)

        if self.fine_tune_file:
            with open("fine_tune_data.json", "a") as f:
                for data in dict_resp:
                    training_data = {
                        "messages": [
                            {"role": "user", "content": data["prompt"]},
                            {"role": "assistant", "content": data["expected"]},
                        ]
                    }
                    json.dump(training_data, f)

        load_extracted_points_to_sql_server(
            self.azure_config.get("schema_name"),
            {d["name"]: d["response"] for d in dict_resp},
            file_name,
        )
        print(f"Done with {file_name}, check response folder and Azure DB!")

    def start(self, load_embed, step_load, out):
        proc_start = time.time()
        proc_docs = 0

        with open("config_prompts.json") as f:
            datapoints_prompts = json.load(f)["datapoints_prompts"]
            for datapoint in datapoints_prompts:
                datapoint["question"] = (
                    f"DATAPOINT: {datapoint['slug_name']}\nDESCRIPTION: {datapoint['desc']}\nQUESTION: {datapoint['question']}"
                )
                self.datapoint_prompts.append(datapoint)

        print("Creating schema and tables...")
        create_schema_tables(
            self.azure_config.get("schema_name"), self.datapoint_prompts
        )

        f_out = os.getcwd() + f"\\resp_{out}\\"
        if not os.path.exists(f_out):
            os.makedirs(f_out)

        processed_docs = []
        if self.azure_config.get("keep_responses"):
            processed_docs = get_processed_file_names(
                self.azure_config.get("schema_name")
            )
            print(f"Don't worry! kept sql table with {len(processed_docs)} records")

        if load_embed:
            print("Adding new embeddings...")
            if step_load:
                print("Used step_load, running embeddings firsts")
                self.embedder.sync_load_embeddings()
            else:
                print("Streaming to dp infer...")
                for resp in self.embedder.gen_load_embeddings():
                    proc_docs += 1
                    self.run_dp_inference(f_out, resp["filename"], out)
        else:
            print("Skip adding new embeddings...")

        if step_load:
            print("Starting to run inference jobs...")
            document_files = self.embedder.get_all_embedded()

            for file_name in document_files:
                if file_name in processed_docs:
                    print(f"{file_name} response found in db, skipping")
                    continue
                proc_docs += 1
                file_name = file_name.split("\\")[-1]
                try:
                    self.run_dp_inference(f_out, file_name, out)
                except Exception as e:
                    print(traceback.format_exc())
                    print(f"{file_name} Error while processing: ", e)

        print("📄 Total docs processed:", proc_docs)
        print("⏲️ Total time taken:", time.time() - proc_start)

    def load_expected_responses(self):
        """Loads expected responses file, expects a column named slug_name"""
        self.expected = {}
        with open("data_points_manual_extract.csv") as f:
            csv_data = csv.DictReader(f)
            for row in csv_data:
                name = row.pop("slug_name")
                row.pop("Report Section")
                self.expected[name] = row

    def get_expected(self, file, dp_name):
        """Helper function to get the expected response for a given datapoint"""
        return self.expected[dp_name].get(file, "")
