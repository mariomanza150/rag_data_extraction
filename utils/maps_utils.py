# rag_pipeline/utils/maps_utils.py
import logging
from typing import Optional, Tuple
import os
import requests

logger = logging.getLogger(__name__)

def get_lat_lng(location_name: str, api_key: Optional[str] = None) -> Optional[Tuple[float, float]]:
    """
    Get latitude and longitude for a given location using Google Maps API.
    
    Args:
        location_name (str): The name or address of the location.
        api_key (Optional[str], optional): API key for Google Maps. Defaults to None.
    
    Returns:
        Optional[Tuple[float, float]]: Tuple of (latitude, longitude) if found, otherwise None.
    """
    if not location_name:
        raise ValueError("Location name cannot be empty.")
    if api_key is None:
        api_key = os.getenv("MAPS_KEY")
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": location_name, "key": api_key}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.exception("Error fetching data from Google Maps API")
        raise RuntimeError(f"Error fetching data from Google Maps API: {e}")
    if data.get("status") == "OK":
        location = data["results"][0]["geometry"]["location"]
        return location["lat"], location["lng"]
    elif data.get("status") == "ZERO_RESULTS":
        return None
    else:
        raise RuntimeError(f"Google Maps API error: {data.get('status')}")
