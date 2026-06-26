from __future__ import annotations

import json
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import pdfplumber


ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = ROOT / "Tomtat-paper" / "TomTat.pdf"
DOCX_SOURCE = Path(r"E:\KL-word.docx")
SCRATCH = ROOT / ".agent" / "scratch" / "tomtat_word_compare"
DOCX_COPY = SCRATCH / "KL-word.copy.docx"

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def docx_cell_text(cell: ET.Element) -> str:
    parts: list[str] = []
    for t in cell.findall(".//w:t", NS):
        if t.text:
            parts.append(t.text)
    return normalize_space("".join(parts))


def extract_docx(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    body = root.find("w:body", NS)
    blocks: list[dict] = []
    paragraphs: list[str] = []
    tables: list[list[list[str]]] = []
    if body is None:
        return {"text": "", "blocks": [], "tables": []}
    for child in body:
        if child.tag == f"{{{NS['w']}}}p":
            text = normalize_space("".join(t.text or "" for t in child.findall(".//w:t", NS)))
            if text:
                paragraphs.append(text)
                blocks.append({"type": "p", "text": text})
        elif child.tag == f"{{{NS['w']}}}tbl":
            rows: list[list[str]] = []
            for tr in child.findall("./w:tr", NS):
                row = [docx_cell_text(tc) for tc in tr.findall("./w:tc", NS)]
                if any(row):
                    rows.append(row)
            if rows:
                tables.append(rows)
                flat = " | ".join(" ; ".join(row) for row in rows[:8])
                blocks.append({"type": "table", "text": flat, "rows": rows})
    return {"text": "\n".join(paragraphs), "blocks": blocks, "tables": tables}


def extract_pdf(path: Path) -> dict:
    pages: list[dict] = []
    blocks: list[dict] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
            text = text.replace("\x00", "")
            pages.append({"page": i, "text": text})
            for para in re.split(r"\n\s*\n|\n(?=[A-ZĐÂĂÊÔƠƯ0-9][^a-z]{0,20})", text):
                para = normalize_space(para)
                if para:
                    blocks.append({"type": "page_text", "page": i, "text": para})
    return {"text": "\n\n".join(p["text"] for p in pages), "pages": pages, "blocks": blocks}


NUM_RE = re.compile(
    r"(?<![\w])(?:\d{1,3}(?:[,.]\d{3})+|\d+(?:[,.]\d+)?)(?:\s*%|\s*(?:--|-|–|—|to)\s*\d{2,4})?",
    re.UNICODE,
)


KEYWORDS = [
    "quan sát",
    "doanh nghiệp",
    "2005",
    "2016",
    "train",
    "val",
    "test",
    "accuracy",
    "f1",
    "auc",
    "roc",
    "macro",
    "weighted",
    "dmf",
    "lstm",
    "tlstm",
    "graphsage",
    "gat",
    "ig",
    "hy",
    "distressed",
    "92",
    "93",
    "94",
    "8,680",
    "8680",
    "1,623",
    "1623",
]


def contexts(blocks: list[dict], label: str) -> list[dict]:
    found: list[dict] = []
    for idx, block in enumerate(blocks):
        text = block.get("text", "")
        low = text.lower()
        nums = NUM_RE.findall(text)
        if nums or any(k in low for k in KEYWORDS):
            found.append(
                {
                    "source": label,
                    "index": idx,
                    "page": block.get("page"),
                    "type": block.get("type"),
                    "nums": nums,
                    "text": text,
                }
            )
    return found


def compact_key(text: str) -> str:
    low = text.lower()
    low = re.sub(r"[,\.]", "", low)
    tokens = re.findall(r"[a-zà-ỹđ]+|\d+%?|\d+", low)
    important = [t for t in tokens if t in KEYWORDS or re.match(r"\d", t)]
    return " ".join(important[:20])


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOCX_SOURCE, DOCX_COPY)

    pdf = extract_pdf(PDF_PATH)
    docx = extract_docx(DOCX_COPY)

    (SCRATCH / "tomtat_pdf_text.txt").write_text(pdf["text"], encoding="utf-8")
    (SCRATCH / "kl_word_text.txt").write_text(docx["text"], encoding="utf-8")
    (SCRATCH / "kl_word_tables.json").write_text(json.dumps(docx["tables"], ensure_ascii=False, indent=2), encoding="utf-8")

    pdf_contexts = contexts(pdf["blocks"], "pdf")
    docx_contexts = contexts(docx["blocks"], "docx")
    (SCRATCH / "numeric_contexts.json").write_text(
        json.dumps({"pdf": pdf_contexts, "docx": docx_contexts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "pdf_pages": len(pdf["pages"]),
        "pdf_blocks": len(pdf["blocks"]),
        "docx_blocks": len(docx["blocks"]),
        "docx_tables": len(docx["tables"]),
        "pdf_numeric_contexts": len(pdf_contexts),
        "docx_numeric_contexts": len(docx_contexts),
        "files": {
            "pdf_text": str(SCRATCH / "tomtat_pdf_text.txt"),
            "docx_text": str(SCRATCH / "kl_word_text.txt"),
            "docx_tables": str(SCRATCH / "kl_word_tables.json"),
            "numeric_contexts": str(SCRATCH / "numeric_contexts.json"),
        },
    }

    # Surface likely comparable snippets by a rough keyword/number key.
    by_key: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"pdf": [], "docx": []})
    for item in pdf_contexts:
        key = compact_key(item["text"])
        if key:
            by_key[key]["pdf"].append(item)
    for item in docx_contexts:
        key = compact_key(item["text"])
        if key:
            by_key[key]["docx"].append(item)
    overlaps = {k: v for k, v in by_key.items() if v["pdf"] and v["docx"]}
    (SCRATCH / "rough_overlaps.json").write_text(json.dumps(overlaps, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["rough_overlap_groups"] = len(overlaps)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
