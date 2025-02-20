# rag_pipeline/utils/regex_parsing.py
"""
Utility functions for regex-based post-processing.
Provides a function to extract numeric values with an expected unit from text.
"""
import re

def parse_numeric_with_unit(text: str, unit: str) -> str:
    """
    Extracts a numeric value followed by the expected unit from the text.
    Returns a string formatted as "number unit" if found; otherwise, returns None.
    """
    pattern = rf"(\d+(\.\d+)?)[\s-]*({re.escape(unit)})"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        number = match.group(1)
        unit_found = match.group(3)
        return f"{number} {unit_found}"
    return None
