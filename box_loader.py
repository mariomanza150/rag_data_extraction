import logging
import os
import subprocess
import time

import pandas as pd
from boxsdk import CCGAuth, Client

from utils import is_report

start_time = time.time()

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler(f"{__name__}.log", mode="a", encoding="utf-8"))


class AzureBoxLoader:
    def __init__(self, loaded_files_path, docs_to_fetch):
        logger.info("Starting Azure Box stroage Document loader...")
        box_client_id = os.getenv("BOX_CLIENT_ID")
        box_client_secret = os.getenv("BOX_CLIENT_SECRET")
        box_user = os.getenv("BOX_USER")

        # Initialize Box client
        auth = CCGAuth(
            client_id=box_client_id,
            client_secret=box_client_secret,
            user=box_user,
            enterprise_id="",
        )

        self.box_client = Client(auth)

        self.loaded_files_path = loaded_files_path
        self.docs_to_fetch = docs_to_fetch
        self.fetched = 0
        self.get_previously_loaded()

    def load_documents(self, folder_ids=[]):

        for root_folder_id in folder_ids:
            # Root folder ID for Box
            try:
                return self.traverse_folder(root_folder_id)
            except Exception as e:
                logger.exception("AzureBoxLoaderERR")
                raise e

    def get_previously_loaded(self):
        try:
            self.previously_loaded = pd.read_csv("loaded_file_list.csv")
        except FileNotFoundError:
            logger.warning("loaded_file_list.csv not found, creating new file.")
            self.previously_loaded = pd.DataFrame()

        if self.previously_loaded.shape[0] > 0:
            self.folder_list = self.previously_loaded["box_directory"].values.tolist()
            logger.info(f"Found {len(self.folder_list)} previously loaded folders")
        else:
            self.folder_list = []

        self.file_list = []

    def save_progress(self):
        df = pd.DataFrame(self.file_list)
        out_file = pd.concat([self.previously_loaded, df])
        try:
            out_file.to_csv("loaded_file_list.csv")
        except Exception:
            logger.exception("AzureBoxLoaderERR - Progress save")
            time.sleep(2)
            out_file.to_csv("loaded_file_list.csv")

    def traverse_folder(self, folder_id, directory=None):
        if self.fetched >= self.docs_to_fetch:
            return self.file_list
        if directory in self.folder_list:
            logger.info("Skipping traversed dir", directory)
            return self.file_list

        folder_items = [
            item for item in self.box_client.folder(folder_id=folder_id).get_items()
        ]

        for item in folder_items:
            if item.type == "folder":
                new_directory = f"{directory}/{item.name}" if directory else item.name
                logger.info(f"Processing {new_directory}")

                # Use recursion to process child folders
                if new_directory not in self.folder_list:
                    self.traverse_folder(item.id, directory=new_directory)
            elif item.type == "file":
                if is_report(item.name):
                    self.save_file(directory, item, item.name)

        # after checking all folder items, mark as searched
        if directory:
            self.file_list.append(
                {"azure_name": "", "box_directory": directory, "box_name": ""}
            )
            self.save_progress()

        return self.file_list

    def save_file(self, directory, file, file_name):
        if file_name.endswith(".wpd") or file_name.endswith(".rpt"):
            with open(file_name, "wb") as f:
                file.download_to(f)
            local_pdf = self.convert_wpd_to_pdf(file_name, self.loaded_files_path)
            file_name = os.path.basename(local_pdf)
        else:
            with open(f"{self.loaded_files_path}/{file_name}", "wb") as f:
                file.download_to(f)
            file_name = file.name
        self.file_list.append(
            {
                "azure_name": file_name,
                "box_directory": directory,
                "box_name": file.name,
            }
        )
        self.save_progress()
        self.fetched = self.fetched + 1
        logger.info(f"Fetched {self.fetched} of {self.docs_to_fetch}")
        logger.info(f"📩 Saved {file_name}")

    def convert_wpd_to_pdf(self, wp_file):
        command_line_exe = "C:\\Program Files\\LibreOffice\\program\\soffice.exe"
        subprocess.run(
            [
                command_line_exe,
                "--headless",
                "--convert-to",
                "pdf",
                wp_file,
                "--outdir",
                self.converted_output_dir,
            ]
        )
        return os.path.join(
            self.converted_output_dir, os.path.splitext(wp_file)[0] + ".pdf"
        )
