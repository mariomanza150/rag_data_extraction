# rag_pipeline/core/loaders/box_loader.py
import logging
import os
import subprocess
import time
from typing import Any, Dict, List

import pandas as pd
from boxsdk import CCGAuth, Client

from rag_pipeline.utils.storage_utils import is_report

logger = logging.getLogger(__name__)

class AzureBoxLoader:
    """
    Loader for documents from Box storage integrated with Azure.
    """
    def __init__(self, loaded_files_path: str, docs_to_fetch: int, config: Dict[str, Any]) -> None:
        """
        Initialize AzureBoxLoader.
        
        Args:
            loaded_files_path (str): Local directory path to save files.
            docs_to_fetch (int): Number of documents to fetch.
            config (Dict[str, Any]): Configuration dictionary containing Box credentials.
        """
        logger.info("Starting Azure Box storage Document loader...")
        self.box_client_id = config.get("BOX_CLIENT_ID")
        self.box_client_secret = config.get("BOX_CLIENT_SECRET")
        self.box_user = config.get("BOX_USER")

        auth = CCGAuth(
            client_id=self.box_client_id,
            client_secret=self.box_client_secret,
            user=self.box_user,
            enterprise_id="",
        )
        self.box_client = Client(auth)
        self.loaded_files_path = loaded_files_path
        self.docs_to_fetch = docs_to_fetch
        self.fetched = 0
        self.get_previously_loaded()

    def load_documents(self, folder_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Load documents from specified Box folder IDs.
        
        Args:
            folder_ids (List[str]): List of Box folder IDs.
        
        Returns:
            List[Dict[str, Any]]: List of file metadata dictionaries.
        """
        file_list: List[Dict[str, Any]] = []
        for root_folder_id in folder_ids:
            try:
                file_list.extend(self.traverse_folder(root_folder_id))
            except Exception as e:
                logger.exception("Error in load_documents")
                raise e
        return file_list

    def get_previously_loaded(self) -> None:
        """
        Retrieve previously loaded file information.
        """
        try:
            self.previously_loaded = pd.read_csv("loaded_file_list.csv")
        except FileNotFoundError:
            logger.warning("loaded_file_list.csv not found, creating new file.")
            self.previously_loaded = pd.DataFrame()

        if not self.previously_loaded.empty:
            self.folder_list = self.previously_loaded["box_directory"].values.tolist()
            logger.info(f"Found {len(self.folder_list)} previously loaded folders")
        else:
            self.folder_list = []
        self.file_list: List[Dict[str, Any]] = []

    def save_progress(self) -> None:
        """
        Save progress of loaded files to CSV.
        """
        df = pd.DataFrame(self.file_list)
        out_file = pd.concat([self.previously_loaded, df])
        try:
            out_file.to_csv("loaded_file_list.csv", index=False)
        except Exception:
            logger.exception("Error saving progress in AzureBoxLoader")
            time.sleep(2)
            out_file.to_csv("loaded_file_list.csv", index=False)

    def traverse_folder(self, folder_id: str, directory: str = None) -> List[Dict[str, Any]]:
        """
        Traverse a Box folder recursively to fetch files.
        
        Args:
            folder_id (str): Box folder ID.
            directory (str, optional): Directory path representation.
        
        Returns:
            List[Dict[str, Any]]: List of file metadata.
        """
        if self.fetched >= self.docs_to_fetch:
            return self.file_list
        if directory and directory in self.folder_list:
            logger.info(f"Skipping traversed dir {directory}")
            return self.file_list

        folder_items = list(self.box_client.folder(folder_id=folder_id).get_items())

        for item in folder_items:
            if item.type == "folder":
                new_directory = f"{directory}/{item.name}" if directory else item.name
                logger.info(f"Processing {new_directory}")
                if new_directory not in self.folder_list:
                    self.traverse_folder(item.id, directory=new_directory)
            elif item.type == "file":
                if is_report(item.name):
                    self.save_file(directory, item, item.name)

        if directory:
            self.file_list.append({"azure_name": "", "box_directory": directory, "box_name": ""})
            self.save_progress()
        return self.file_list

    def save_file(self, directory: str, file: Any, file_name: str) -> None:
        """
        Save a file from Box to local storage, converting if necessary.
        
        Args:
            directory (str): Box directory name.
            file (Any): Box file object.
            file_name (str): Name of the file.
        """
        try:
            if file_name.lower().endswith((".wpd", ".rpt")):
                local_path = os.path.join(os.getcwd(), file_name)
                with open(local_path, "wb") as f:
                    file.download_to(f)
                local_pdf = self.convert_wpd_to_pdf(file_name, self.loaded_files_path)
                file_name = os.path.basename(local_pdf)
            else:
                local_path = os.path.join(self.loaded_files_path, file_name)
                with open(local_path, "wb") as f:
                    file.download_to(f)
                file_name = file.name
            self.file_list.append({
                "azure_name": file_name,
                "box_directory": directory,
                "box_name": file.name,
            })
            self.save_progress()
            self.fetched += 1
            logger.info(f"Fetched {self.fetched} of {self.docs_to_fetch}")
            logger.info(f"Saved {file_name}")
        except Exception as e:
            logger.exception(f"Error saving file {file_name}: {e}")

    def convert_wpd_to_pdf(self, wp_file: str, converted_output_dir: str) -> str:
        """
        Convert a WPD file to PDF using LibreOffice.
        
        Args:
            wp_file (str): The WPD file name.
            converted_output_dir (str): Output directory for converted PDF.
        
        Returns:
            str: Path to the converted PDF file.
        """
        command_line_exe = "C:\\Program Files\\LibreOffice\\program\\soffice.exe"
        subprocess.run([
            command_line_exe,
            "--headless",
            "--convert-to",
            "pdf",
            wp_file,
            "--outdir",
            converted_output_dir,
        ], check=True)
        pdf_path = os.path.join(converted_output_dir, os.path.splitext(wp_file)[0] + ".pdf")
        return pdf_path
