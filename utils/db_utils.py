# utils/db_utils.py
import json
import logging
from typing import Any, Dict, List

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

def get_db_engine(db_config: Dict[str, str]) -> Engine:
    server = db_config.get("DB_SERVER")
    database = db_config.get("DB_NAME")
    username = db_config.get("DB_USERNAME")
    password = db_config.get("DB_PASSWORD")
    connection_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+18+for+SQL+Server"
    return create_engine(connection_str)

def get_processed_file_names(engine: Engine, schema_name: str) -> List[str]:
    query = text(f"SELECT azure_filename FROM {schema_name}.report_data")
    with engine.connect() as conn:
        result = conn.execute(query)
        names = [row[0] for row in result]
        logger.debug(f"Retrieved processed file names: {names}")
        return names

def create_schema_tables(engine: Engine, schema_name: str, datapoints: List[Dict[str, Any]]) -> None:
    with engine.connect() as conn:
        try:
            with conn.begin():
                logger.info(f"Creating schema: {schema_name}")
                create_query = text(f"CREATE SCHEMA {schema_name}")
                conn.execute(create_query)
                if False:
                    conn.execute(text(f"DROP TABLE {schema_name}.report_data;"))
                create_data_table = text(
                    f"""CREATE TABLE {schema_name}.report_data (
                        azure_filename varchar(255) UNIQUE,
                        report_type varchar(255),
                        insert_timestamp DATETIME,
                        raw_extractions varchar(max)
                    );"""
                )
                conn.execute(create_data_table)
                logger.info(f"Schema and table created in {schema_name}.")
        except Exception as e:
            error_message = str(e)
            if "There is already an object named" in error_message:
                logger.warning("Schema and Table already exist! Response overwriting will occur if flag set.")
            else:
                logger.exception("Error creating schema tables.")
                raise

def write_project_data(engine: Engine, schema_name: str, data: Dict[str, Any], table_name: str = "report_data") -> Dict[str, Any]:
    with engine.connect() as conn:
        with conn.begin():
            delete_query = text(
                f"DELETE FROM {schema_name}.{table_name} WHERE azure_filename= :azure_filename"
            )
            conn.execute(delete_query, {"azure_filename": data["azure_filename"]})
            columns = ", ".join(data.keys())
            placeholders = ", ".join([f":{key}" for key in data.keys()])
            insert_query = text(
                f"INSERT INTO {schema_name}.{table_name} ({columns}) VALUES ({placeholders})"
            )
            clean_data = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    response = value.get(key, json.dumps(value))
                    clean_data[key] = json.dumps(response) if isinstance(response, dict) else f"{str(response)}"
                elif isinstance(value, list):
                    clean_data[key] = ",".join(value)
                elif key == "insert_timestamp":
                    clean_data[key] = value
                else:
                    clean_data[key] = str(value)
                try:
                    clean_data[key] = clean_data[key].encode()[:8000].decode()
                except AttributeError:
                    pass
            conn.execute(insert_query, clean_data)
            logger.info(f"Wrote data for file {data.get('azure_filename')}")
    return data

def load_extracted_points_to_sql_server(engine: Engine, schema_name: str, data: Dict[str, Any], file_name: str) -> None:
    report_data = {
        "azure_filename": file_name,
        "insert_timestamp": pd.Timestamp.now(),
        "raw_extractions": json.dumps(data)
    }
    write_project_data(engine, schema_name, report_data)
    logger.info(f"Wrote {file_name} responses data to SQL Server.")
