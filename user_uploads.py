"""populate data.json with user uploaded pdfs
# fields that are extracted by heuristics:
# - paper (title)
# - year
# - url

# fields that may be extracted by NLP in the future:
# - tasks
# - datasets
# - model
# - derived_from
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

OUTPUT_DIR = "knowledge_graphs/uploaded"
ARXIV_PATTERN = re.compile(
    r"arXiv:(\d{4}\.\d{4,5})(v\d+)?\s*\[([^\]]+)\]\s*(\d{1,2}\s+\w+\s+\d{4})",
    re.IGNORECASE
) # e.g., arXiv:2004.12345v2 [cs.CL] 00 Apr 2025

# ! IMPORTANT: extract pdf text in main pipeline and pass first 2 pages to this module

def extract_title(text: str) -> Optional[str]:
    """Heuristic: title usually appears at top
    2 non-empty lines with reasonable length"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = ""
    for line in lines[:2]:
        if 20 < len(line) < 200 and not line.lower().startswith("abstract"):
            title += line + " "
    return title.strip()

def extract_arxiv_meta(text: str) -> Optional[str]:
    """heuristic: arxiv metadata (e.g., arXiv:1234.12345v2 [cs.CL] 00 Apr 2025)
    usually appears at bottom of first page when extracting text
    This should extract year and arxiv id to build url
    """
    match = ARXIV_PATTERN.search(text)
    if not match:
        return None

    arxiv_id = match.group(1)
    date_str = match.group(4)

    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", date_str)
    year = int(year_match.group(0)) if year_match else None

    return {
        "url": f"https://arxiv.org/pdf/{arxiv_id}",
        "year": year,
        "arxiv_id": arxiv_id
    }

def build_metadata(pdf_path: Path, text: str) -> Dict:
    """build heuristic-based metadata json"""
    arxiv_meta = extract_arxiv_meta(text)

    metadata = {
        "paper": extract_title(text),
        "pdf_path": str(pdf_path),
        "year": arxiv_meta["year"] if arxiv_meta else None,
        "url": arxiv_meta["url"] if arxiv_meta else None,
        "metadata_source": {
            "paper": "heuristic",
            "year": "arxiv",
            "url": "arxiv",
        }
    }

    if arxiv_meta:
        metadata["arxiv_id"] = arxiv_meta["arxiv_id"]

    return metadata

def save_to_json(metadata: Dict, output_dir: str = OUTPUT_DIR):
    """save metadata to json file"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "uploaded_data.json")
    
    data = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                content = json.load(f)
                if isinstance(content, list):
                    data = content
                else:
                    data = [content]
        except json.JSONDecodeError:
            pass

    data.append(metadata)

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)