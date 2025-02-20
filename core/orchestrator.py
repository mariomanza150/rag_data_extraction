# core/orchestrator.py
import asyncio
import csv
import json
import os
import time
import logging
from typing import Any, Dict, List

from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI

from core.embeddings.embedding_loader import EmbeddingLoader
from core.llm.llm_manager import LLModelManager
from core.embeddings.vector_store import VectorStore
from utils.db_utils import create_schema_tables, get_processed_file_names, load_extracted_points_to_sql_server
from utils.maps_utils import get_lat_lng
from utils.regex_parsing import parse_numeric_with_unit

logger = logging.getLogger(__name__)

class DataPointManager:
    def __init__(self, llm_manager: LLModelManager, es_config: Dict[str, Any]) -> None:
        self.llm_manager = llm_manager
        self.datapoint_template = self.llm_manager.get_datapoint_template()
        self.datapoint_llm = self.llm_manager.get_datapoint_model()
        self.es_config = es_config

    def build_prompt(self, db_results: List[Any], query: str) -> str:
        if isinstance(self.datapoint_llm, OllamaLLM):
            template = ChatPromptTemplate.from_template(self.datapoint_template)
        elif isinstance(self.datapoint_llm, ChatOpenAI):
            template = PromptTemplate.from_template(self.datapoint_template)
        context_text = "\n".join([doc.page_content for doc, _ in db_results])
        return template.format(context=context_text, question=query)

class ResultWriter:
    def __init__(self, fine_tune_file: bool, log_prompts: bool, log_expected: bool) -> None:
        self.fine_tune_file = fine_tune_file
        self.log_prompts = log_prompts
        self.log_expected = log_expected

    def write_results(self, out: str, responses: List[List[Any]], all_responses: List[Dict[str, Any]], file_name: str, azure_config: Dict[str, Any]) -> None:
        clean_filename = file_name.replace(".pdf", "").replace("-Signed", "").strip()
        resp_dir = os.path.join(os.getcwd(), f"resp_{out}")
        os.makedirs(resp_dir, exist_ok=True)
        resp_filename = os.path.join(resp_dir, f"{clean_filename}.{out}")
        try:
            if out == "csv":
                with open(resp_filename, "a", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(["datapoint", "actual_response", "expected_response", "reference"])
                    for row in responses:
                        writer.writerow(row)
            elif out == "json":
                with open(resp_filename, "a", encoding="utf-8") as f:
                    json.dump(all_responses, f, indent=2)

            if self.fine_tune_file:
                with open("fine_tune_data.json", "a", encoding="utf-8") as f:
                    for data in all_responses:
                        training_data = {
                            "messages": [
                                {"role": "user", "content": data.get("prompt", "")},
                                {"role": "assistant", "content": data.get("expected", "")},
                            ]
                        }
                        json.dump(training_data, f)
            if azure_config.get("use_db"):
                load_extracted_points_to_sql_server(
                    azure_config.get("db_engine"),
                    azure_config.get("schema_name"),
                    {d["name"]: {"value": d["response"], "reference": d.get("reference", "")} for d in all_responses},
                    file_name,
                )
            logger.info(f"Done with {file_name}, check responses!")
        except Exception as e:
            logger.exception(f"Error writing results for {file_name}: {e}")

class Orchestrator:
    def __init__(self, log_prompts: bool, log_expected: bool, fine_tune_file: bool, es_config: Dict[str, Any], el_config: Dict[str, Any], az_config: Dict[str, Any]) -> None:
        self.llm_manager = LLModelManager()
        self.data_point_manager = DataPointManager(self.llm_manager, es_config)
        self.result_writer = ResultWriter(fine_tune_file, log_prompts, log_expected)
        self.es_config = es_config
        self.vector_store = VectorStore(es_config, self.llm_manager.get_embedding_model())
        self.embedder = EmbeddingLoader(self.llm_manager.get_embedding_model(), **el_config)
        self.azure_config = az_config
        self.log_prompt = log_prompts
        self.log_expected = log_expected
        self.overwrite = True
        self.datapoint_prompts: List[Dict[str, Any]] = []

    async def create_prompt(self, file_name: str, datapoint: Dict[str, Any]) -> Dict[str, Any]:
        try:
            filters = {"source": file_name}
            if datapoint.get("section"):
                filters["section"] = datapoint["section"]
            conf = self.es_config.copy()
            conf.pop("clear_db", None)
            vs = VectorStore(conf, self.llm_manager.get_embedding_model())
            results = []
            for search in datapoint["search"].split("/"):
                keywords = " ".join(search.split(","))
                results += vs.search_chunks(keywords, filters, top_k=3)
            if results:
                query = datapoint["question"]
                prompt = self.data_point_manager.build_prompt(results, query)
                return {"dp": datapoint, "prompt": prompt}
            else:
                return {"dp": {"datapoint": datapoint.get("slug_name", "")}, "response": "no related chunks found"}
        except RuntimeError as e:
            logger.exception(f"Error with embeddings for {file_name}: {e}")
            self.vector_store.delete_file_embed(file_name)
            return {"dp": {"datapoint": datapoint.get("slug_name", "")}, "response": "error during prompt creation"}

    async def process_datapoints(self, file_name: str, datapoints_prompts: List[Dict[str, Any]]) -> List[Any]:
        async with asyncio.TaskGroup() as tg:
            for datapoint in datapoints_prompts:
                tg.create_task(self.create_prompt(file_name, datapoint))
                await asyncio.sleep(0)
        return await asyncio.gather(*[self.create_prompt(file_name, dp) for dp in datapoints_prompts], return_exceptions=True)

    def run_dp_inference(self, fold_out: str, file_name: str, out: str) -> None:
        clean_filename = file_name.replace(".pdf", "").replace("-Signed", "").strip()
        resp_filename = os.path.join(fold_out, f"{clean_filename}.{out}")
        if os.path.exists(resp_filename) and not self.overwrite:
            logger.info(f"Skipping, {clean_filename} response file found")
            return

        logger.info(f"Searching for datapoints from: {clean_filename}")
        logger.info("Formatting prompts...")
        dpoints = asyncio.run(self.process_datapoints(file_name, self.datapoint_prompts))
        not_found, found_data, prompts_list = [], [], []
        for dp in dpoints:
            if isinstance(dp, dict) and "prompt" in dp:
                found_data.append(dp)
                prompts_list.append(dp["prompt"])
            else:
                not_found.append(dp)
        logger.info(f"Formatting prompts complete: {len(found_data)} of {len(self.datapoint_prompts)} found")
        logger.info("Getting responses...")
        start = time.time()
        raw_responses = []
        if isinstance(self.data_point_manager.datapoint_llm, OllamaLLM):
            raw_responses = self.llm_manager.sync_local_batch_job(self.data_point_manager.datapoint_llm, prompts_list)
        elif isinstance(self.data_point_manager.datapoint_llm, ChatOpenAI):
            for idx, p in enumerate(prompts_list):
                raw_responses.append((idx, self.llm_manager.make_openai_call(self.data_point_manager.datapoint_llm, p)))
        logger.info(f"Time taken for {len(raw_responses)} responses: {time.time() - start}")

        responses, dict_resp = [], []
        for resp in raw_responses:
            dp = found_data[resp[0]]
            name = dp["dp"].get("slug_name")
            prompt_text = dp.get("prompt") if self.log_prompt else ""
            expected = self.get_expected(clean_filename, name) if self.log_expected else ""
            resp_content = resp[1] if isinstance(resp[1], str) else resp[1].content
            try:
                json_resp = json.loads(resp_content)
                response_text = json_resp.get("extracted", json_resp.get(name, json_resp))
            except json.decoder.JSONDecodeError:
                response_text = resp_content

            unit = ""
            lower_question = dp["dp"].get("question", "").lower()
            lower_desc = dp["dp"].get("desc", "").lower()
            if "psf" in lower_question or "psf" in lower_desc:
                unit = "psf"
            elif "feet" in lower_question or "feet" in lower_desc:
                unit = "feet"
            elif "inches" in lower_question or "inches" in lower_desc:
                unit = "inches"
            if unit:
                parsed = parse_numeric_with_unit(response_text, unit)
                if parsed is not None:
                    response_text = parsed
                    try:
                        numeric_value = float(parsed.split()[0])
                        if (unit == "psf" and numeric_value > 1e6) or (unit in ["feet", "inches"] and numeric_value > 1000):
                            logger.warning(f"Suspected outlier for {name} in {file_name}: {response_text}")
                            extra_flag = "suspected_outlier"
                        else:
                            extra_flag = ""
                    except Exception:
                        extra_flag = ""
                else:
                    response_text = "Parsing failed"
                    extra_flag = ""
            else:
                extra_flag = ""
            reference = dp["dp"].get("page") or "N/A"
            responses.append([name, response_text, expected, prompt_text, reference])
            dict_item = {
                "name": name,
                "response": response_text,
                "expected": expected,
                "prompt": prompt_text,
                "reference": reference
            }
            if extra_flag:
                dict_item["flag"] = extra_flag
            dict_resp.append(dict_item)
        self.result_writer.write_results(out, responses, dict_resp, file_name, self.azure_config)

    def start(self, load_embed: bool, step_load: bool, out: str) -> None:
        proc_start = time.time()
        proc_docs = 0
        with open("config/config_prompts.json", encoding="utf-8") as f:
            datapoints_prompts = json.load(f)["datapoints_prompts"]
            for datapoint in datapoints_prompts:
                datapoint["question"] = (
                    f"DATAPOINT: {datapoint['slug_name']}\n"
                    f"DESCRIPTION: {datapoint['desc']}\n"
                    f"QUESTION: {datapoint['question']}"
                )
                self.datapoint_prompts.append(datapoint)
        processed_docs = []
        f_out = os.path.join(os.getcwd(), f"resp_{out}")
        os.makedirs(f_out, exist_ok=True)
        if self.azure_config.get("use_db"):
            logger.info("Creating schema and tables...")
            from utils.db_utils import get_db_engine
            db_engine = get_db_engine(self.azure_config.get("db_config", {}))
            self.azure_config["db_engine"] = db_engine
            create_schema_tables(db_engine, self.azure_config.get("schema_name"), self.datapoint_prompts)
            if self.azure_config.get("keep_responses"):
                processed_docs = get_processed_file_names(db_engine, self.azure_config.get("schema_name"))
                logger.info(f"SQL table has {len(processed_docs)} records")
        if load_embed:
            logger.info("Adding new embeddings...")
            if step_load:
                logger.info("Using step_load, running embeddings first")
                self.embedder.sync_load_embeddings()
            else:
                logger.info("Streaming to datapoint inference...")
                for resp in self.embedder.gen_load_embeddings():
                    proc_docs += 1
                    self.run_dp_inference(f_out, resp["filename"], out)
        else:
            logger.info("Skipping new embeddings...")
        if step_load:
            logger.info("Starting inference jobs...")
            document_files = self.embedder.get_all_embedded()
            for file_name in document_files:
                if file_name in processed_docs:
                    logger.info(f"{file_name} response found in db, skipping")
                    continue
                proc_docs += 1
                file_name_only = file_name.split(os.sep)[-1]
                try:
                    self.run_dp_inference(f_out, file_name_only, out)
                except Exception as e:
                    logger.exception(f"{file_name_only} Error while processing: {e}")
        logger.info(f"Total docs processed: {proc_docs}")
        logger.info(f"Total time taken: {time.time() - proc_start}")

    def load_expected_responses(self) -> None:
        self.expected: Dict[str, Dict[str, str]] = {}
        try:
            with open("data_points_manual_extract.csv", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.pop("slug_name")
                    row.pop("Report Section", None)
                    self.expected[name] = row
            logger.info("Expected responses loaded.")
        except Exception as e:
            logger.exception(f"Error loading expected responses: {e}")

    def get_expected(self, file: str, dp_name: str) -> str:
        return self.expected.get(dp_name, {}).get(file, "")
