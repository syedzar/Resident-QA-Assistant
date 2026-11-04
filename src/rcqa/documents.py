"""Turn raw source files into clean text units, keeping the location needed for citations.

* PDFs (RNAO, PHO)     -> one row per page   (citation: "p. 37")
* e-Laws text (.txt)   -> one row per document; split into sections at chunking time (stage 5)
* openFDA label JSON   -> one row per label section (citation: "Boxed Warning")
Pure Python, no Spark, so it is unit-tested directly.
"""
from __future__ import annotations

import io
import json
import re

# openFDA keys that are metadata or tables, not readable label text
_SKIP_KEYS = {"openfda", "set_id", "id", "effective_time", "version", "spl_product_data_elements",
              "package_label_principal_display_panel", "spl_unclassified_section", "spl_medguide",
              "spl_patient_package_insert", "spl_indexing_data_elements"}


def clean_text(text: str) -> str:
    """Normalise whitespace and common PDF artefacts without changing wording."""
    text = text.replace("­", "")                 # soft hyphens
    text = re.sub(r"-\n(?=[a-z])", "", text)          # words hyphenated across lines
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_pages(data: bytes) -> list[tuple[int, str]]:
    """[(page_number starting at 1, text), ...]. Pages with no extractable text are kept as ''."""
    import logging

    from pypdf import PdfReader  # imported here so the rest of the module works without pypdf

    logging.getLogger("pypdf").setLevel(logging.ERROR)  # font-encoding notices are noise here
    reader = PdfReader(io.BytesIO(data))
    return [(i + 1, clean_text(p.extract_text() or "")) for i, p in enumerate(reader.pages)]


def section_title(key: str) -> str:
    """'warnings_and_cautions' -> 'Warnings and Cautions'"""
    small = {"and", "or", "of", "in", "for", "to"}
    words = key.split("_")
    return " ".join(w if (i and w in small) else w.capitalize() for i, w in enumerate(words))


def label_sections(label_json: str) -> list[tuple[str, str]]:
    """[(section title, text), ...] in the label's own order, text sections only."""
    label = json.loads(label_json)
    out = []
    for key, value in label.items():
        if key in _SKIP_KEYS or key.endswith("_table") or not isinstance(value, list):
            continue
        text = clean_text("\n".join(v for v in value if isinstance(v, str)))
        if text:
            out.append((section_title(key), text))
    return out
