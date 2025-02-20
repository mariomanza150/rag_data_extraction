# rag_pipeline/utils/storage_utils.py
import logging
import re
from typing import List, Any
from math import ceil

from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)

def get_azure_blob_client(connection_str: str) -> BlobServiceClient:
    """
    Create an Azure BlobServiceClient using the provided connection string.
    
    Args:
        connection_str (str): Azure Blob connection string.
    
    Returns:
        BlobServiceClient: The blob service client.
    """
    return BlobServiceClient.from_connection_string(connection_str)

def is_report(file_name: str) -> bool:
    """
    Determine if a file name corresponds to a report based on specific patterns.
    
    Args:
        file_name (str): Name of the file.
    
    Returns:
        bool: True if file name matches report criteria, else False.
    """
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
        logger.info(f"{file_name} matched criteria.")
        return True
    else:
        logger.info(f"{file_name} not processed, no report indicator in filename.")
        return False

def filter_by_latest(file_names: List[str]) -> List[str]:
    """
    Filter a list of file names to get the latest version based on naming convention.
    
    Args:
        file_names (List[str]): List of file names.
    
    Returns:
        List[str]: Filtered list with the latest file names.
    """
    exp = r"DN(?P<project>(\d{3,})|([\d,]{3,}))(-(?P<major>\d{2,})|(\.(?P<minor>\d{2,})-(\d{2,})))-R(?P<rev>\d{1,})"
    groups = {}
    for f in file_names:
        match = re.search(exp, f)
        id_key = f
        f_low = f.lower()
        if match:
            id_key = "DN" + match.group("project")
            major = int(match.group("major")) if match.group("major") is not None else int(match.group(8))
            minor = 0 if match.group("minor") is None else int(match.group("minor"))
            rev = 0 if match.group("rev") is None else int(match.group("rev"))
            revised = 1 if "revise" in f_low else 0
            signed = 1 if "sign" in f_low else 0
            final = 1 if "final" in f_low else 0
            sub = True
            if id_key in groups:
                sub = False
                latest = groups[id_key]
                if latest["major"] == major:
                    if latest["minor"] == minor:
                        if latest["rev"] == rev:
                            latest_flags = latest["revised"] + latest["signed"] + latest["final"]
                            current_flags = revised + signed + final
                            if latest_flags < current_flags:
                                sub = True
                        elif latest["rev"] < rev:
                            sub = True
                    elif latest["minor"] < minor:
                        sub = True
                elif latest["major"] < major:
                    sub = True
            groups[id_key] = ({"file_name": f, "major": major, "minor": minor, "rev": rev, "revised": revised, "signed": signed, "final": final} if sub else groups[id_key])
        else:
            groups[id_key] = {"file_name": f}
    return [v["file_name"] for v in groups.values()]

async def iter_over_async(ait, loop) -> Any:
    """
    Asynchronously iterate over an asynchronous iterator using a loop executor.
    
    Args:
        ait: Asynchronous iterator.
        loop: Event loop.
    
    Returns:
        Any: Next item from the asynchronous iterator.
    """
    ait = ait.__aiter__()
    async def get_next():
        try:
            obj = await ait.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None
    while True:
        done, obj = await loop.run_in_executor(None, get_next)
        if done:
            break
        yield obj

def chunk(lst, size, lst2=None):
    num_chunks = ceil(len(lst) / size)
    if lst2 is None:
        return [lst[i * size: i * size + size] for i in range(num_chunks)]
    else:
        return [
            (lst[i * size: i * size + size], lst2[i * size: i * size + size])
            for i in range(num_chunks)
        ]
