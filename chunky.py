import json
import re

from langchain_core.prompts import PromptTemplate
from rapidfuzz import fuzz

PROMPT_TEMPLATE = """
SUBTITLES: ```{options}```

CONTENT: ```{context}```

---

Given the possible SUBTITLES, which one would you pick based on the CONTENT?
Strictly respond with an option from the SUBTITLES list.
Only return the SUBTITLE name variable value.
"""


class Chunky:
    """This class is responsible for cleaning and tagging the file content."""

    def __init__(self, llm):
        with open("config_sections.json") as f:
            self.possible_sections = json.load(f)["report_sections"]

        self.llm = llm

    def clean_text(self, text):
        """A function that replaces all repeating spaces in string and removes empty lines"""

        text = re.sub(" +", " ", text.lower())
        text = re.sub(r"\s+(a|an|and|the)\s+", " ", text)
        # try to extract type, name and address
        text = re.sub(r"(?P<original>(?P<type>^[\s\w]*((invest)|(assess))\w*)\s*(?P<name>[ \w.]*)(\s|,)(?P<address>[\s\w]*,[ \w]*).*)\s*prepared for", r"\g<original>. report type:\g<type>.report name/filing:\g<name>.site address:\g<address>. prepared for", text)
        # remove client address and insert possible client name
        text = re.sub(r"prepared for:\s*([ \w]*)\s*([\s\w,]*)\s*project", r"prepared for: \g<1>. client name: \g<1>. project", text)
        # try to extract project number and report date
        text = re.sub(r"project no\.*(.*)\s*([ \w,]*)", r"project no.\g<1> \g<2>. project number:\g<1>. report date:\g<2>", text)
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"\.{2,}", " ", text)
        text = re.sub(r"\\+", " ", text)
        # remove ctl address
        text = re.sub(r"([\s\w]*)\|([\s\w,]*)\|\s*phone:\s*([\s\d-]*)\s*\|\s*fax:\s*([\s\d-]*)", " phone fax ", text)
        text = text.split("\n")
        text = [
            line.replace("\x01\x02\x03\x04\x05\x06\x07\x08\t", "")
            .replace("\u0002", "")
            .replace("\u201c", "")
            .replace("\u201d", "")
            .replace("\u00b1", "")
            .strip()
            for line in text
            if line.replace("\x01\x02\x03\x04\x05\x06\x07\x08\t", "")
            .replace("\u0002", "")
            .replace("\u201c", "")
            .replace("\u201d", "")
            .replace("\u00b1", "")
            .strip()
            != ""
        ]
        text = " ".join(text)
        return text

    async def process_documents(self, filename, documents):
        doc_count = len(documents)

        if self.llm:
            sections = [self.clean_text(s.get("slug")) for s in self.possible_sections]
            prompts = []
            template = PromptTemplate.from_template(PROMPT_TEMPLATE)
            for doc in documents:
                doc.page_content = self.clean_text(doc.page_content)
                prompts.append(
                    template.format(context=doc.page_content, options=sections)
                )

            print(f"Running inference on {doc_count} pages, for {filename}")
            async for response in self.llm.abatch_as_completed(prompts):
                doc = documents[response[0]]
                title = max(
                    sections, key=lambda name: fuzz.ratio(response[1].content, name)
                )
                text = f"SECTION: {title}\nCONTENT: {doc.page_content}\nSECTION:{title}"
                doc.metadata["section"] = title
                yield text, doc.metadata
        else:
            for doc in documents:
                yield self.clean_text(doc.page_content), doc.metadata
