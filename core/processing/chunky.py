# core/processing/chunky.py
import json
import logging
import re
from typing import AsyncGenerator, List, Tuple, Any

from langchain_core.prompts import PromptTemplate
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
SUBTITLES: ```{options}```

CONTENT: ```{context}```

---

Given the possible SUBTITLES, which one would you pick based on the CONTENT?
Strictly respond with an option from the SUBTITLES list.
Only return the SUBTITLE name variable value.
"""


class Chunky:
    """
    Responsible for cleaning, chunking and tagging file content with section labels.
    It first attempts to detect headings and known synonyms; if none are found, it uses an LLM.
    """
    def __init__(self, llm: Any) -> None:
        with open("config/config_sections.json", encoding="utf-8") as f:
            self.possible_sections = json.load(f)["report_sections"]
        self.llm = llm

    def clean_text(self, text: str) -> str:
        text = re.sub(" +", " ", text.lower())
        text = re.sub(r"\s+(a|an|and|the)\s+", " ", text)
        text = re.sub(r"(?P<original>(?P<type>^[\s\w]*((invest)|(assess))\w*)\s*(?P<name>[ \w.]*)(\s|,)(?P<address>[\s\w]*,[ \w]*).*)\s*prepared for",
                      r"\g<original>. report type:\g<type>.report name/filing:\g<name>.site address:\g<address>. prepared for", text)
        text = re.sub(r"prepared for:\s*([ \w]*)\s*([\s\w,]*)\s*project",
                      r"prepared for: \g<1>. client name: \g<1>. project", text)
        text = re.sub(r"project no\.*(.*)\s*([ \w,]*)",
                      r"project no.\g<1> \g<2>. project number:\g<1>. report date:\g<2>", text)
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"\.{2,}", " ", text)
        text = re.sub(r"\\+", " ", text)
        text = re.sub(r"([\s\w]*)\|([\s\w,]*)\|\s*phone:\s*([\s\d-]*)\s*\|\s*fax:\s*([\s\d-]*)", " phone fax ", text)
        lines = text.split("\n")
        cleaned_lines = [
            line.replace("\x01\x02\x03\x04\x05\x06\x07\x08\t", "")
                .replace("\u0002", "")
                .replace("\u201c", "")
                .replace("\u201d", "")
                .replace("\u00b1", "")
                .strip()
            for line in lines
            if line.replace("\x01\x02\x03\x04\x05\x06\x07\x08\t", "")
                  .replace("\u0002", "")
                  .replace("\u201c", "")
                  .replace("\u201d", "")
                  .replace("\u00b1", "")
                  .strip() != ""
        ]
        cleaned_text = " ".join(cleaned_lines)
        return cleaned_text

    def detect_heading(self, text: str) -> str:
        """
        Detects a section heading in text by matching section slugs, names, or their synonyms.
        Returns the detected section slug if found; otherwise, returns an empty string.
        """
        for section in self.possible_sections:
            slug = section.get("slug", "").lower()
            name = section.get("name", "").lower()
            synonyms = [s.lower() for s in section.get("other_names", [])]
            if re.search(r"\b" + re.escape(slug) + r"\b", text):
                return slug
            if re.search(r"\b" + re.escape(name) + r"\b", text):
                return slug
            for syn in synonyms:
                if re.search(r"\b" + re.escape(syn) + r"\b", text):
                    return slug
        return ""

    async def process_documents(self, filename: str, documents: List[Any]) -> AsyncGenerator[Tuple[str, dict], None]:
        """
        Processes documents to assign section labels. Uses rule-based heading detection first;
        if no heading is found and an LLM is available, falls back to LLM classification.
        """
        for doc in documents:
            cleaned = self.clean_text(doc.page_content)
            doc.page_content = cleaned
            detected = self.detect_heading(cleaned)
            if detected:
                doc.metadata["section"] = detected
                text = f"SECTION: {detected}\nCONTENT: {cleaned}\nSECTION: {detected}"
                yield text, doc.metadata
            elif self.llm:
                sections = [self.clean_text(s.get("slug", "")) for s in self.possible_sections]
                template = PromptTemplate.from_template(PROMPT_TEMPLATE)
                prompt = template.format(context=cleaned, options=",".join(sections))
                logger.info(f"LLM fallback for document {filename} page {doc.metadata.get('page')}")
                response = await self.llm.acall(prompt)
                chosen = max(sections, key=lambda s: fuzz.ratio(response.content, s))
                doc.metadata["section"] = chosen
                text = f"SECTION: {chosen}\nCONTENT: {cleaned}\nSECTION: {chosen}"
                yield text, doc.metadata
            else:
                yield cleaned, doc.metadata
