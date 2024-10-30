import json
import os
import re

import pandas as pd
import requests
from azure.storage.blob import BlobServiceClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

server = os.getenv("DB_SERVER")
database = os.getenv("DB_NAME")
username = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")
engine = create_engine(
    f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+18+for+SQL+Server"
)
maps_key = os.getenv("MAPS_KEY")


def is_report(file_name):
    # pattern = r"(R\d+|Report\d+|RPT)"
    # pattern = r"-[1]\d{2}-R\d+-Signed"
    pattern1 = r"[1]\d{2}"
    pattern2 = r"R\d+"
    pattern3 = r"Signed"
    extension = r"\.(pdf|LTR|RPT|WPD|SF)$"
    if (
        re.search(extension, file_name, re.IGNORECASE)
        and re.search(pattern1, file_name)
        and re.search(pattern2, file_name)
        and re.search(pattern3, file_name)
    ):
        print(f"{file_name} matched criteria.")
        return True
    else:
        print(f"{file_name} not processed, no report indicator in filename.")
        return False


def get_azure_blob_client():
    connect_str = os.getenv("AZURE_BLOB_CONNECTION_STR")
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    return blob_service_client


def get_processed_file_names(schema_name):
    query = text(f"SELECT azure_filename FROM {schema_name}.report_data")
    with engine.connect() as conn:
        response = list(conn.execute(query))
        return [row[0] for row in response]


def get_lat_lng(location_name: str, api_key: str = maps_key) -> tuple:
    """
    Get the latitude and longitude coordinates of a location using its name.

    Args:
        location_name (str): The name of the location to get coordinates for.
        api_key (str): Your Google Maps API key.

    Returns:
        tuple: A tuple containing latitude and longitude coordinates.

    Raises:
        ValueError: If the location name is empty.
        RuntimeError: If the Google Maps API returns an error.
    """
    if not location_name:
        raise ValueError("Location name cannot be empty.")

    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": location_name, "key": api_key}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Error fetching data from Google Maps API: {e}")
    if data["status"] == "OK":
        location = data["results"][0]["geometry"]["location"]
        lat = location["lat"]
        lng = location["lng"]
        return lat, lng
    elif data["status"] == "ZERO_RESULTS":
        return None, None
    else:
        raise RuntimeError(f"Google Maps API error: {data['status']}")


def create_schema_tables(schema_name, datapoints):
    with engine.connect() as conn:
        # Start a transaction
        try:
            with conn.begin():
                print(f"Schema name: {schema_name}")
                create_query = text(f"CREATE SCHEMA {schema_name}")
                conn.execute(create_query)

                if False:  # uncomment to drop table
                    conn.execute(text(f"DROP TABLE {schema_name}.report_data;"))

                create_data_table = text(f"""CREATE TABLE {schema_name}.report_data
                                        (azure_filename varchar(255) UNIQUE,
                                        report_type varchar(255),
                                        insert_timestamp DATETIME,
                                        {' varchar(8000), '.join([d['slug_name'] for d in datapoints])} varchar(255));""")

                conn.execute(create_data_table)
        except Exception as e:
            if "There is already an object named" in e._sql_message():
                print(
                    "Schema and Table already exist! Response overwriting will occur if flag set"
                )
            else:
                raise e


def load_extracted_points_to_sql_server(schema_name, data, file_name):
    report_data = {
        "azure_filename": file_name,
        "insert_timestamp": pd.Timestamp.now(),
    } | data
    write_project_data(schema_name, engine, report_data, "report_data")
    print(f"Wrote {file_name} responses data to SQL Server.")


def write_project_data(schema_name, engine, data, table_name):
    """
    Writes data to the 'report_data' table in the Azure SQL database.
    If a row with the same project_numebr exists, it deletes that row first.

    :param engine: SQLAlchemy engine
    :param data: Dictionary containing the data to be written
    """

    with engine.connect() as conn:
        # Start a transaction
        with conn.begin():
            # Delete existing rows with the same project_number
            delete_query = text(
                f"DELETE FROM {schema_name}.{table_name} WHERE azure_filename= :azure_filename"
            )
            conn.execute(delete_query, {"azure_filename": data["azure_filename"]})

            # Prepare dynamic columns for insertion
            columns = ", ".join(data.keys())
            placeholders = ", ".join([f":{key}" for key in data.keys()])

            # Insert the new data
            insert_query = text(
                f"INSERT INTO {schema_name}.{table_name} ({columns}) VALUES ({placeholders})"
            )

            clean_data = {}
            for key, value in data.items():
                if type(value) is dict:
                    response = value.get(key, json.dumps(value))
                    if type(response) is dict:
                        clean_data[key] = json.dumps(response)
                    else:
                        clean_data[key] = f'"{str(response)}"'
                elif type(value) is list:
                    clean_data[key] = '"' + ",".join(value) + '"'
                elif key in ["insert_timestamp"]:
                    clean_data[key] = value
                else:
                    clean_data[key] = str(value)

                # limit values to varchar size for responses (for strings)
                try:
                    clean_data[key] = clean_data[key].encode()[:8000].decode()
                except AttributeError:
                    pass

            result = conn.execute(insert_query, clean_data)
            return result


def filter_by_latest(file_names):
    exp = r"DN(?P<project>(\d{3,})|([\d,]{3,}))(-(?P<major>\d{2,})|(\.(?P<minor>\d{2,})-(\d{2,})))-R(?P<rev>\d{1,})"
    groups = {}
    for f in file_names:
        match = re.search(exp, f)
        id = f
        f_low = f.lower()
        if match:
            id = "DN" + match.group("project")
            major = (
                int(match.group(8))
                if match.group("major") is None
                else int(match.group("major"))
            )
            minor = 0 if match.group("minor") is None else int(match.group("minor"))
            rev = 0 if match.group("rev") is None else int(match.group("rev"))
            revised = 1 if "revise" in f_low else 0
            signed = 1 if "sign" in f_low else 0
            final = 1 if "final" in f_low else 0
            sub = True
            if id in groups:
                sub = False
                latest = groups[id]
                if latest["major"] == major:
                    if latest["minor"] == minor:
                        if latest["rev"] == rev:
                            latest_flags = (
                                latest["revised"] + latest["signed"] + latest["final"]
                            )
                            current_flags = revised + signed + final
                            if latest_flags < current_flags:
                                sub = True
                        elif latest["rev"] < rev:
                            sub = True
                    elif latest["minor"] < minor:
                        sub = True
                elif latest["major"] < major:
                    sub = True

            groups[id] = (
                {
                    "file_name": f,
                    "major": major,
                    "minor": minor,
                    "rev": rev,
                    "revised": revised,
                    "signed": signed,
                    "final": final,
                }
                if sub
                else latest
            )

        else:
            groups[id] = {"file_name": f}

    return [v["file_name"] for v in groups.values()]


def iter_over_async(ait, loop):
    ait = ait.__aiter__()

    async def get_next():
        try:
            obj = await ait.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None

    while True:
        done, obj = loop.run_until_complete(get_next())
        if done:
            break
        yield obj
