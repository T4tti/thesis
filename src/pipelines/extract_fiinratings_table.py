#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FiinRatings Table Extractor - "Bảng biểu 01: Các chỉ số phân tích chính"
=========================================================================
Strategy:
  1. Dùng pdfplumber extract words (với x/y coordinates) từng trang
  2. Tìm trang + vị trí dòng có "Bảng biểu 01"
  3. Lấy vùng bảng ngay sau heading đó
  4. Detect cột từ dòng header (năm: 2021A, 2022A...)
  5. Assign mỗi số vào cột gần nhất theo x-center
  6. Export DataFrame / CSV / JSON

How to Run:
  python src/pipelines/extract_fiinratings_table.py [PDF_PATH]

Expected Output:
  data/processed/<stem>_bang_bieu_01.csv
"""

import re
import sys
import json
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = (
    ROOT
    / "data/external/fiinratings_output/pdfs"
    / "[VI]_FiinRatings_CMX_Public_Announcement_20230331.pdf"
)
OUTPUT_DIR = ROOT / "data/processed"

# ── Patterns ──────────────────────────────────────────────────────────────────
YEAR_PATTERN = re.compile(r"^(20\d{2}[AF]?)$", re.IGNORECASE)
TABLE_HEADING_KWS = ["bảng biểu 01", "bang bieu 01", "các chỉ số phân tích chính"]
TABLE_END_KWS = ["nguồn:", "source:", ": fiinratings", "nguon:"]


def import_pdfplumber():
    try:
        import pdfplumber
        return pdfplumber
    except ImportError:
        print("Thiếu pdfplumber. Chạy: pip install pdfplumber")
        sys.exit(1)


def parse_vn_number(s: str) -> Optional[float]:
    """
    Parse số theo format VN hoặc EN:
      '2,094' → 2094.0   (thousands separator dạng US)
      '1.089' → 1089.0   (thousands separator dạng VN)
      '0.62'  → 0.62
      '0,62'  → 0.62
    """
    s = s.strip().rstrip("%")
    if not s:
        return None

    # Detect format dựa vào số ký tự sau separator cuối cùng
    if "," in s and "." in s:
        # Có cả hai → tìm cái nào là decimal
        if s.rindex(",") > s.rindex("."):
            # Dấu phẩy là decimal: 1.089,50 → 1089.50
            s = s.replace(".", "").replace(",", ".")
        else:
            # Dấu chấm là decimal: 1,089.50
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) == 3 and parts[0].isdigit() and parts[1].isdigit():
            # Vd: 1,089 → 1089 (thousands separator)
            s = s.replace(",", "")
        else:
            # Vd: 0,62 → 0.62 (decimal)
            s = s.replace(",", ".")
    elif "." in s:
        parts = s.split(".")
        if len(parts) == 2 and len(parts[1]) == 3 and parts[0].isdigit() and parts[1].isdigit():
            # Vd: 1.089 → 1089 (VN thousands separator)
            s = s.replace(".", "")
        # else: 0.62, 4.09, v.v. → giữ nguyên

    try:
        return float(s)
    except ValueError:
        return None


def extract_table_from_pdf(pdf_path: Path) -> Optional[pd.DataFrame]:
    """
    Main extraction function sử dụng word-level coordinates.
    """
    pdfplumber = import_pdfplumber()

    print(f"[PDF] {pdf_path.name}")
    print("=" * 64)

    target_page = None

    # ── BƯỚC 1: Tìm trang chứa "Bảng biểu 01" ────────────────────────────
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            txt = (page.extract_text() or "").lower()
            if any(kw in txt for kw in TABLE_HEADING_KWS):
                target_page = page
                print(f"[1] Bảng tìm thấy trên trang: {page_num}")
                break

        if target_page is None:
            print("[!] Không tìm thấy 'Bảng biểu 01' trong file PDF.")
            return None

        # BƯỚC 2: Extract words có tọa độ
        words = target_page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=False,
        )

    print(f"[2] Words trên trang: {len(words)}")

    # ── BƯỚC 3: Cluster words → lines theo y-coordinate ────────────────────
    LINE_TOL = 4  # px tolerance cho cùng dòng

    lines_map: dict[int, list[dict]] = {}
    for w in words:
        k = round(w["top"] / LINE_TOL) * LINE_TOL
        lines_map.setdefault(k, []).append(w)

    lines = []
    for y in sorted(lines_map.keys()):
        row_words = sorted(lines_map[y], key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in row_words)
        lines.append({"y": y, "text": text, "words": row_words})

    # ── BƯỚC 4: Locate heading và header năm ──────────────────────────────
    heading_idx = None
    for i, line in enumerate(lines):
        if any(kw in line["text"].lower() for kw in TABLE_HEADING_KWS):
            heading_idx = i
            print(f"[3] Heading tại line #{i}: '{line['text'][:70]}'")
            break

    if heading_idx is None:
        print("[!] Không locate heading trong word-list.")
        return None

    # Tìm header năm (dòng có >= 2 tokens khớp YEAR_PATTERN)
    year_x_map: dict[str, float] = {}
    header_idx = None

    for i in range(heading_idx + 1, min(heading_idx + 10, len(lines))):
        line = lines[i]
        yr_words = [w for w in line["words"] if YEAR_PATTERN.match(w["text"])]
        if len(yr_words) >= 2:
            header_idx = i
            for w in yr_words:
                cx = (w["x0"] + w["x1"]) / 2
                year_x_map[w["text"]] = cx
            print(f"[4] Header tại line #{i}: '{line['text']}'")
            print(f"    Cột → x_center: {year_x_map}")
            break

    if not year_x_map:
        print("[!] Không tìm thấy header năm trong 10 dòng sau heading.")
        return None

    years = list(year_x_map.keys())
    # Ranh giới x phân tách label vs dữ liệu
    label_x_boundary = min(year_x_map.values()) - 35

    # ── BƯỚC 5: Parse dòng dữ liệu ────────────────────────────────────────
    print(f"\n[5] Parsing (label_x_boundary = {label_x_boundary:.1f}px):")
    header_str = f"    {'Chỉ tiêu':<35}" + "".join(f"{yr:>10}" for yr in years)
    print(header_str)
    print("    " + "-" * (35 + 10 * len(years)))

    records: dict[str, dict] = {}
    COL_TOL = 50  # px - max distance để assign word vào cột

    for i in range(header_idx + 1, len(lines)):
        line = lines[i]
        text = line["text"].strip()
        if not text:
            continue

        # Kết thúc bảng
        if any(text.lower().startswith(kw) for kw in TABLE_END_KWS):
            print(f"    → [End of table: '{text[:40]}']")
            break

        row_words = line["words"]

        # Tách label-words (bên trái) và data-words (bên phải)
        label_ws = [w for w in row_words if w["x1"] <= label_x_boundary + 20]
        data_ws  = [w for w in row_words if w["x0"] >= label_x_boundary - 5]

        label = " ".join(w["text"] for w in sorted(label_ws, key=lambda w: w["x0"])).strip()
        label = re.sub(r"^[▪▸•\-–\s]+", "", label).strip()

        # Parse và assign data values → cột gần nhất
        row_vals: dict[str, Optional[float]] = {yr: None for yr in years}
        for dw in data_ws:
            val = parse_vn_number(dw["text"])
            if val is None:
                continue
            cx = (dw["x0"] + dw["x1"]) / 2
            dists = sorted((abs(cx - xc), yr) for yr, xc in year_x_map.items())
            best_dist, best_yr = dists[0]
            if best_dist <= COL_TOL:
                row_vals[best_yr] = val

        has_data = any(v is not None for v in row_vals.values())
        if label and has_data:
            records[label] = row_vals
            val_str = "".join(
                f"{str(row_vals[yr]) if row_vals[yr] is not None else 'NULL':>10}"
                for yr in years
            )
            print(f"    {label:<35}{val_str}")

    if not records:
        print("[!] Không extract được dòng dữ liệu nào.")
        return None

    # ── BƯỚC 6: Tạo DataFrame ─────────────────────────────────────────────
    df = pd.DataFrame(records, index=years).T
    df.index.name = "Chỉ tiêu"
    df.columns.name = "Năm"

    return df


def main():
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        pdf_path = DEFAULT_PDF

    if not pdf_path.exists():
        print(f"File không tồn tại: {pdf_path}")
        sys.exit(1)

    df = extract_table_from_pdf(pdf_path)

    if df is None:
        print("Extraction thất bại.")
        sys.exit(1)

    print("\n" + "=" * 64)
    print("BANG BIEU 01 - KET QUA:")
    print("=" * 64)
    print(df.to_string())

    # Lưu CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[\[\]]", "", pdf_path.stem)
    output_csv = OUTPUT_DIR / f"{stem}_bang_bieu_01.csv"
    df.to_csv(output_csv, encoding="utf-8-sig")
    print(f"\nDa luu: {output_csv}")

    # JSON
    json_out = {}
    for col in df.columns:
        json_out[col] = {
            k: (float(v) if pd.notna(v) else None)
            for k, v in df[col].items()
        }
    print("\nJSON:")
    print(json.dumps(json_out, ensure_ascii=False, indent=2))

    return df


if __name__ == "__main__":
    main()
