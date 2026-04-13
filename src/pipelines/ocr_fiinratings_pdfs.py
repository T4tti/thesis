#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Extraction Pipeline - FiinRatings PDFs → CSV
=================================================
Mục tiêu:
  - Đọc các file PDF báo cáo XHTN từ data/external/fiinratings_output/pdfs/
  - Kết hợp metadata từ credit_ratings.csv (scraper đã thu thập)
  - Trích xuất các chỉ số tài chính bằng pdfplumber + regex
  - Xuất ra CSV có schema tương tự corporateCreditRatingWithFinancialRatios.csv
  - Các trường thiếu → NULL

How to Run:
  pip install pdfplumber pymupdf pandas
  python src/pipelines/ocr_fiinratings_pdfs.py

Expected Output:
  data/processed/fiinratings_ocr_extracted.csv
  (147 hàng metadata + dữ liệu tài chính extracted từ PDF, NULL nếu không tìm thấy)
"""

import re
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

# ── Cấu hình logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Đường dẫn ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = ROOT / "data" / "external" / "fiinratings_output" / "pdfs"
META_CSV = ROOT / "data" / "external" / "fiinratings_output" / "credit_ratings.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "fiinratings_ocr_extracted.csv"

# ── Schema output (tương tự corporateCreditRatingWithFinancialRatios.csv) ──────
OUTPUT_COLUMNS = [
    "Rating Agency",
    "Corporation",
    "Rating",
    "Rating Date",
    "CIK",
    "Binary Rating",
    "SIC Code",
    "Sector",
    "Ticker",
    "Current Ratio",
    "Long-term Debt / Capital",
    "Debt/Equity Ratio",
    "Gross Margin",
    "Operating Margin",
    "EBIT Margin",
    "EBITDA Margin",
    "Pre-Tax Profit Margin",
    "Net Profit Margin",
    "Asset Turnover",
    "ROE - Return On Equity",
    "Return On Tangible Equity",
    "ROA - Return On Assets",
    "ROI - Return On Investment",
    "Operating Cash Flow Per Share",
    "Free Cash Flow Per Share",
    # Các trường bổ sung đặc thù FiinRatings VN
    "Outlook",
    "Rating Type",
    "Industry",
    "ISIN",
    "Source PDF",
    "OCR Confidence",
]

# ── Binary rating mapping (Vietnam scale: D→AAA) ──────────────────────────────
# Investment Grade (AAA, AA+, AA, AA-, A+, A, A-) → 1
# Below Investment Grade → 0
INVESTMENT_GRADE = {"AAA", "AA+", "AA", "AA-", "A+", "A", "A-"}

# ── Sector mapping từ ngành nghề Việt nam ─────────────────────────────────────
SECTOR_MAP = {
    "dịch vụ tài chính": "Money",
    "ngân hàng": "Money",
    "chứng khoán": "Money",
    "bảo hiểm": "Money",
    "doanh nghiệp phi tài chính": "Other",
    "bất động sản": "Other",
    "xây dựng": "Other",
    "năng lượng": "Enrgy",
    "điện": "Enrgy",
    "công nghệ": "BusEq",
    "viễn thông": "Telcm",
    "y tế": "Hlth",
    "sản xuất": "Manuf",
    "thực phẩm": "NoDur",
    "bán lẻ": "Shops",
}


def map_sector(industry_str: str) -> str:
    """Map tên ngành Việt Nam sang mã sector như dataset gốc."""
    if not industry_str or pd.isna(industry_str):
        return "NULL"
    lower = industry_str.lower()
    for key, val in SECTOR_MAP.items():
        if key in lower:
            return val
    return "Other"


def binary_rating(rating: str) -> int:
    """Chuyển mức xếp hạng → 1 (Investment Grade) / 0 (Speculative)."""
    if not rating or pd.isna(rating):
        return 0
    rating_clean = str(rating).strip().upper()
    # Remove outlook suffix if any: e.g., "BBB+" → "BBB+"
    return 1 if rating_clean in INVESTMENT_GRADE else 0


# ── Tên các chỉ tiêu tài chính theo pattern ngôn ngữ VN + EN ──────────────────
FINANCIAL_PATTERNS: Dict[str, List[str]] = {
    # Current Ratio / Tỷ số thanh toán hiện hành
    "Current Ratio": [
        r"(?:hệ số|tỷ số|current ratio|tỷ lệ)[\s:]*.{0,30}?thanh\s*toán\s*(?:hiện\s*hành|ngắn\s*hạn|nhanh).{0,20}?([\d.,]+)",
        r"current\s*ratio[\s:=]*([\d.,]+)",
        r"(?:CR|hslq)\s*[=:]\s*([\d.,]+)",
    ],
    # Debt/Equity Ratio
    "Debt/Equity Ratio": [
        r"(?:tỷ lệ|hệ số)[\s:]*.{0,20}?(?:nợ\/vốn|D/E|nợ trên vốn chủ).{0,20}?([\d.,]+)",
        r"(?:D/E|debt[\s/]equity)\s*(?:ratio)?[\s:=]*([\d.,]+)",
        r"(?:tỷ lệ nợ|nợ\/vốn chủ)\s*[=:]\s*([\d.,]+)",
    ],
    # Long-term Debt / Capital
    "Long-term Debt / Capital": [
        r"(?:nợ dài hạn\/vốn|long.?term\s*debt.*capital)\s*[=:]?\s*([\d.,]+)",
        r"nợ dài hạn\s*/\s*(?:tổng vốn|tổng nguồn vốn)\s*[=:]?\s*([\d.,]+)",
    ],
    # Gross Margin / Biên lợi nhuận gộp
    "Gross Margin": [
        r"(?:biên lợi nhuận gộp|gross\s*(?:profit\s*)?margin|tỷ suất lợi nhuận gộp)\s*[=:]?\s*([\d.,]+)\s*%?",
        r"(?:GPM)\s*[=:]?\s*([\d.,]+)",
    ],
    # Operating Margin / Biên lợi nhuận hoạt động
    "Operating Margin": [
        r"(?:biên lợi nhuận(?:\s*từ)?(?:\s*hoạt động)?|operating\s*(?:profit\s*)?margin|tỷ suất lợi nhuận hoạt động)\s*[=:]?\s*([\d.,]+)\s*%?",
        r"(?:OPM)\s*[=:]?\s*([\d.,]+)",
    ],
    # EBIT Margin
    "EBIT Margin": [
        r"(?:EBIT\s*margin|biên EBIT|tỷ suất EBIT)\s*[=:]?\s*([\d.,]+)\s*%?",
    ],
    # EBITDA Margin
    "EBITDA Margin": [
        r"(?:EBITDA\s*margin|biên EBITDA|tỷ suất EBITDA)\s*[=:]?\s*([\d.,]+)\s*%?",
    ],
    # Net Profit Margin / Biên lợi nhuận ròng
    "Net Profit Margin": [
        r"(?:biên lợi nhuận (?:ròng|sau thuế)|net\s*(?:profit\s*)?margin|tỷ suất lợi nhuận (?:ròng|sau thuế))\s*[=:]?\s*([\d.,]+)\s*%?",
        r"(?:NPM)\s*[=:]?\s*([\d.,]+)",
    ],
    # Pre-Tax Profit Margin
    "Pre-Tax Profit Margin": [
        r"(?:biên lợi nhuận trước thuế|pre.?tax\s*(?:profit\s*)?margin)\s*[=:]?\s*([\d.,]+)\s*%?",
    ],
    # ROE
    "ROE - Return On Equity": [
        r"(?:ROE|tỷ suất lợi nhuận trên vốn chủ|return\s*on\s*equity)\s*[=:]?\s*([\d.,]+)\s*%?",
        r"(?:lợi nhuận\/vốn chủ)\s*[=:]?\s*([\d.,]+)",
    ],
    # ROA
    "ROA - Return On Assets": [
        r"(?:ROA|tỷ suất lợi nhuận trên tổng tài sản|return\s*on\s*assets)\s*[=:]?\s*([\d.,]+)\s*%?",
    ],
    # ROI
    "ROI - Return On Investment": [
        r"(?:ROI|return\s*on\s*invest)\s*[=:]?\s*([\d.,]+)\s*%?",
    ],
    # Asset Turnover / Vòng quay tổng tài sản
    "Asset Turnover": [
        r"(?:vòng quay tổng tài sản|asset\s*turnover)\s*[=:]?\s*([\d.,]+)",
        r"(?:AT)\s*[=:]?\s*([\d.,]+)",
    ],
    # ROAE / Return On Tangible Equity (dùng làm proxy)
    "Return On Tangible Equity": [
        r"(?:ROAE|return\s*on\s*average\s*equity)\s*[=:]?\s*([\d.,]+)\s*%?",
    ],
}

# ── Table-based extraction patterns ───────────────────────────────────────────
TABLE_COLUMN_ALIASES: Dict[str, List[str]] = {
    "Current Ratio": ["current ratio", "thanh toán hiện hành", "hslq", "thanh toán NH"],
    "Debt/Equity Ratio": ["d/e", "nợ/vốn chủ", "debt/equity"],
    "Gross Margin": ["gross margin", "biên lợi nhuận gộp", "gpm", "lợi nhuận gộp/dt"],
    "Operating Margin": ["operating margin", "biên lợi nhuận hđ", "opm", "lợi nhuận hđ/dt"],
    "EBIT Margin": ["ebit margin", "biên ebit", "ebit/dt"],
    "EBITDA Margin": ["ebitda margin", "biên ebitda", "ebitda/dt"],
    "Net Profit Margin": ["net margin", "biên lợi nhuận ròng", "npm", "lợi nhuận sau thuế/dt"],
    "ROE - Return On Equity": ["roe", "lợi nhuận/vốn chủ"],
    "ROA - Return On Assets": ["roa", "lợi nhuận/tổng ts"],
    "Asset Turnover": ["asset turnover", "vòng quay ts", "at"],
}


def try_import_pdfplumber():
    """Lazy import pdfplumber."""
    try:
        import pdfplumber
        return pdfplumber
    except ImportError:
        logger.warning("pdfplumber chưa cài. Chạy: pip install pdfplumber")
        return None


def try_import_pymupdf():
    """Lazy import pymupdf (fitz)."""
    try:
        import fitz
        return fitz
    except ImportError:
        logger.warning("pymupdf chưa cài. Chạy: pip install pymupdf")
        return None


def clean_number(value: str) -> Optional[float]:
    """Làm sạch chuỗi số → float. Xử lý cả format VN (1.234,56) và EN (1,234.56)."""
    if not value:
        return None
    s = value.strip()
    # Loại bỏ ký tự không phải số
    s = re.sub(r"[%\s]", "", s)
    # Detect format: nếu có dấu phẩy trước dấu chấm → VN format
    if re.search(r"\d{1,3}(?:\.\d{3})+,\d", s):
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")
    elif "," in s:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_text_pdfplumber(pdf_path: Path) -> tuple[str, list]:
    """
    Trích xuất toàn bộ text và bảng từ PDF dùng pdfplumber.
    Returns: (full_text, tables_list)
    """
    pdfplumber = try_import_pdfplumber()
    if pdfplumber is None:
        return "", []

    full_text = ""
    all_tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract text
                txt = page.extract_text() or ""
                full_text += txt + "\n"
                # Extract tables
                tables = page.extract_tables()
                if tables:
                    all_tables.extend(tables)
    except Exception as e:
        logger.warning(f"  pdfplumber lỗi ({pdf_path.name}): {e}")

    return full_text, all_tables


def extract_text_pymupdf(pdf_path: Path) -> str:
    """Fallback: dùng pymupdf để extract text."""
    fitz = try_import_pymupdf()
    if fitz is None:
        return ""
    try:
        doc = fitz.open(str(pdf_path))
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    except Exception as e:
        logger.warning(f"  pymupdf lỗi ({pdf_path.name}): {e}")
        return ""


def extract_from_tables(tables: list) -> Dict[str, Optional[float]]:
    """
    Tìm kiếm chỉ tiêu tài chính trong các bảng đã extract.
    Chiến lược: duyệt qua bảng, tìm hàng có tên chỉ tiêu → lấy giá trị cuối cùng.
    """
    results: Dict[str, Optional[float]] = {}

    for table in tables:
        if not table or len(table) < 2:
            continue

        # Header normalization
        headers = [str(c).lower().strip() if c else "" for c in (table[0] or [])]

        for row in table[1:]:
            if not row:
                continue
            row_label = str(row[0]).lower().strip() if row[0] else ""

            # Tìm match với các chỉ tiêu
            for field, aliases in TABLE_COLUMN_ALIASES.items():
                if field in results:
                    continue
                for alias in aliases:
                    if alias in row_label:
                        # Lấy giá trị: ưu tiên cột cuối cùng không rỗng
                        for val_str in reversed(row[1:]):
                            v = clean_number(str(val_str)) if val_str else None
                            if v is not None:
                                results[field] = v
                                break
                        break

    return results


def extract_from_text(text: str) -> Dict[str, Optional[float]]:
    """
    Tìm kiếm chỉ tiêu tài chính trong text thô bằng regex.
    """
    results: Dict[str, Optional[float]] = {}
    text_lower = text.lower()

    for field, patterns in FINANCIAL_PATTERNS.items():
        if field in results:
            continue
        for pattern in patterns:
            try:
                match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
                if match:
                    val = clean_number(match.group(1))
                    if val is not None:
                        # Sanity check: loại bỏ giá trị vô lý
                        if field.endswith("Margin") or field.startswith("ROE") or field.startswith("ROA"):
                            if -200 <= val <= 200:
                                results[field] = val
                        elif field == "Asset Turnover":
                            if 0 <= val <= 50:
                                results[field] = val
                        else:
                            if -100 <= val <= 100:
                                results[field] = val
                        break
            except re.error as e:
                logger.debug(f"  Regex error ({field}): {e}")

    return results


def process_single_pdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Pipeline trích xuất cho một file PDF.
    Returns dict với các chỉ tiêu tài chính (hoặc None nếu không tìm thấy).
    """
    results: Dict[str, Any] = {}
    confidence_score = 0

    logger.info(f"  → OCR: {pdf_path.name}")

    # Bước 1: Thử pdfplumber (tốt hơn cho text-based PDF)
    full_text, tables = extract_text_pdfplumber(pdf_path)

    # Bước 2: Nếu text rỗng → thử pymupdf
    if len(full_text.strip()) < 100:
        logger.debug(f"    pdfplumber text ngắn, thử pymupdf...")
        full_text = extract_text_pymupdf(pdf_path)

    if len(full_text.strip()) < 50:
        logger.warning(f"    Không extract được text từ {pdf_path.name} (có thể là scan image)")
        results["OCR Confidence"] = 0
        return results

    # Bước 3: Extract từ bảng (độ chính xác cao hơn)
    if tables:
        table_results = extract_from_tables(tables)
        results.update(table_results)
        confidence_score += len(table_results) * 2

    # Bước 4: Extract từ text bằng regex (bổ sung)
    text_results = extract_from_text(full_text)
    for k, v in text_results.items():
        if k not in results and v is not None:
            results[k] = v
            confidence_score += 1

    results["OCR Confidence"] = round(confidence_score / max(len(OUTPUT_COLUMNS) - 10, 1), 2)
    return results


def find_pdf_for_row(row: pd.Series, pdf_dir: Path) -> Optional[Path]:
    """
    Tìm file PDF tương ứng với một hàng metadata từ credit_ratings.csv.
    Dựa vào tên file trong cột PDF_VI.
    """
    pdf_vi = str(row.get("PDF_VI", "")).strip()
    if not pdf_vi:
        return None

    # Lấy tên file từ URL
    pdf_filename = pdf_vi.split("/")[-1]
    # URL decode
    import urllib.parse
    pdf_filename = urllib.parse.unquote(pdf_filename)

    candidate = pdf_dir / pdf_filename
    if candidate.exists():
        return candidate

    # Thử tìm file tương đương (match một phần tên)
    stem = Path(pdf_filename).stem[:30]
    for f in pdf_dir.glob("*.pdf"):
        if stem.lower()[:20] in f.name.lower()[:30]:
            return f

    return None


class FiinRatingsOCRPipeline:
    """
    Pipeline chính: kết hợp metadata CSV + OCR từ PDF
    → xuất CSV có cùng schema với corporateCreditRatingWithFinancialRatios.csv
    """

    def __init__(
        self,
        meta_csv: Path = META_CSV,
        pdf_dir: Path = PDF_DIR,
        output_csv: Path = OUTPUT_CSV,
    ):
        self.meta_csv = meta_csv
        self.pdf_dir = pdf_dir
        self.output_csv = output_csv

    def load_metadata(self) -> pd.DataFrame:
        """Load credit_ratings.csv đã scrape."""
        logger.info(f"Đọc metadata: {self.meta_csv}")
        df = pd.read_csv(self.meta_csv, encoding="utf-8-sig")
        logger.info(f"  Tổng số hàng metadata: {len(df)}")
        return df

    def build_output_row(
        self, meta_row: pd.Series, ocr_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Xây dựng một hàng output theo schema chuẩn.
        Các trường không có → NULL.
        """
        rating_str = str(meta_row.get("Mức xếp hạng", "")).strip()
        rating_date_raw = str(meta_row.get("Ngày", "")).strip()

        # Chuẩn hóa ngày từ DD/MM/YYYY → M/D/YYYY (như dataset gốc)
        try:
            from datetime import datetime
            dt = datetime.strptime(rating_date_raw, "%d/%m/%Y")
            rating_date = dt.strftime("%-m/%-d/%Y") if sys.platform != "win32" else dt.strftime("%#m/%#d/%Y")
        except Exception:
            rating_date = rating_date_raw

        pdf_vi_url = str(meta_row.get("PDF_VI", "")).strip()
        pdf_filename = pdf_vi_url.split("/")[-1] if pdf_vi_url else ""

        row = {
            "Rating Agency": "FiinRatings",
            "Corporation": str(meta_row.get("Tên tổ chức", "")).strip(),
            "Rating": rating_str if rating_str != "nan" else None,
            "Rating Date": rating_date,
            "CIK": None,  # Không có CIK cho công ty Việt Nam
            "Binary Rating": binary_rating(rating_str),
            "SIC Code": None,
            "Sector": map_sector(str(meta_row.get("Ngành", ""))),
            "Ticker": None,
            # Chỉ tiêu tài chính (từ OCR)
            "Current Ratio": ocr_data.get("Current Ratio"),
            "Long-term Debt / Capital": ocr_data.get("Long-term Debt / Capital"),
            "Debt/Equity Ratio": ocr_data.get("Debt/Equity Ratio"),
            "Gross Margin": ocr_data.get("Gross Margin"),
            "Operating Margin": ocr_data.get("Operating Margin"),
            "EBIT Margin": ocr_data.get("EBIT Margin"),
            "EBITDA Margin": ocr_data.get("EBITDA Margin"),
            "Pre-Tax Profit Margin": ocr_data.get("Pre-Tax Profit Margin"),
            "Net Profit Margin": ocr_data.get("Net Profit Margin"),
            "Asset Turnover": ocr_data.get("Asset Turnover"),
            "ROE - Return On Equity": ocr_data.get("ROE - Return On Equity"),
            "Return On Tangible Equity": ocr_data.get("Return On Tangible Equity"),
            "ROA - Return On Assets": ocr_data.get("ROA - Return On Assets"),
            "ROI - Return On Investment": ocr_data.get("ROI - Return On Investment"),
            "Operating Cash Flow Per Share": None,  # Thường không có trong báo cáo VN
            "Free Cash Flow Per Share": None,
            # Trường bổ sung đặc thù VN
            "Outlook": str(meta_row.get("Triển vọng", "")).strip(),
            "Rating Type": str(meta_row.get("Phân loại", "")).strip(),
            "Industry": str(meta_row.get("Ngành", "")).strip(),
            "ISIN": str(meta_row.get("ISIN", "")).strip() or None,
            "Source PDF": pdf_filename,
            "OCR Confidence": ocr_data.get("OCR Confidence", 0),
        }

        # Thay thế giá trị "nan"/"None" trống → None (sẽ thành NULL trong CSV)
        for k, v in row.items():
            if v == "nan" or v == "" or v == "None":
                row[k] = None

        return row

    def run(self) -> pd.DataFrame:
        """Chạy toàn bộ pipeline."""
        logger.info("=" * 60)
        logger.info("FiinRatings OCR Pipeline bắt đầu")
        logger.info("=" * 60)

        # 1. Load metadata
        meta_df = self.load_metadata()

        # 2. Xử lý từng hàng
        output_rows: List[Dict[str, Any]] = []
        pdf_found = 0
        pdf_missing = 0

        for idx, row in meta_df.iterrows():
            corp_name = str(row.get("Tên tổ chức", "")).strip()
            logger.info(f"\n[{idx+1}/{len(meta_df)}] {corp_name}")

            # Tìm PDF tương ứng
            pdf_path = find_pdf_for_row(row, self.pdf_dir)

            if pdf_path and pdf_path.exists():
                pdf_found += 1
                # OCR extraction
                ocr_data = process_single_pdf(pdf_path)
            else:
                pdf_missing += 1
                logger.warning(f"  ⚠ Không tìm thấy PDF cho: {row.get('PDF_VI', 'N/A')}")
                ocr_data = {"OCR Confidence": 0}

            # Build output row
            out_row = self.build_output_row(row, ocr_data)
            output_rows.append(out_row)

        # 3. Tạo DataFrame output
        output_df = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)

        # 4. Thống kê
        logger.info("\n" + "=" * 60)
        logger.info("THỐNG KÊ KẾT QUẢ")
        logger.info("=" * 60)
        logger.info(f"Tổng số hàng : {len(output_df)}")
        logger.info(f"PDF tìm thấy: {pdf_found}")
        logger.info(f"PDF thiếu   : {pdf_missing}")

        # Đếm số giá trị non-null cho mỗi chỉ tiêu tài chính
        financial_cols = [
            "Current Ratio", "Debt/Equity Ratio", "Gross Margin",
            "Operating Margin", "EBIT Margin", "EBITDA Margin",
            "Net Profit Margin", "ROE - Return On Equity", "ROA - Return On Assets",
            "Asset Turnover",
        ]
        logger.info("\nTỷ lệ điền được (non-NULL) per chỉ tiêu:")
        for col in financial_cols:
            if col in output_df.columns:
                n_filled = output_df[col].notna().sum()
                pct = 100 * n_filled / len(output_df)
                logger.info(f"  {col:<40}: {n_filled:3d}/{len(output_df)} ({pct:.1f}%)")

        # 5. Lưu output
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(self.output_csv, index=False, encoding="utf-8-sig", na_rep="NULL")
        logger.info(f"\n✅ Đã lưu: {self.output_csv}")
        logger.info(f"   Shape: {output_df.shape}")

        return output_df


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Kiểm tra dependency
    missing_deps = []
    try:
        import pdfplumber  # noqa: F401
    except ImportError:
        missing_deps.append("pdfplumber")
    try:
        import fitz  # noqa: F401
    except ImportError:
        missing_deps.append("pymupdf")

    if missing_deps:
        logger.warning(f"Các thư viện chưa cài: {', '.join(missing_deps)}")
        logger.warning(f"Cài đặt bằng: pip install {' '.join(missing_deps)}")
        logger.warning("Script sẽ tiếp tục nhưng có thể không extract được text từ PDF scan.")

    pipeline = FiinRatingsOCRPipeline()
    result_df = pipeline.run()

    # Preview kết quả
    print("\n--- Preview 5 hàng đầu ---")
    preview_cols = ["Rating Agency", "Corporation", "Rating", "Rating Date",
                    "Sector", "Gross Margin", "ROE - Return On Equity", "OCR Confidence"]
    available = [c for c in preview_cols if c in result_df.columns]
    print(result_df[available].head(5).to_string(index=False))
