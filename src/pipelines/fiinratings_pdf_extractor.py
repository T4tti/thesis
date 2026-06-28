"""
fiinratings_pdf_extractor.py
----------------------------
Extracts financial indicators and credit ratings from FiinRatings PDFs.
Uses pdfplumber for native text extraction, PaddleOCR for fallback/images,
and Gemini API (google-genai SDK v2) for semantic Key Information Extraction (KIE).

HOW TO RUN:
    1. pip install pdfplumber pandas google-genai pydantic python-dotenv paddleocr
    2. Create .env in src/pipelines/ with: GEMINI_API_KEY=your_key
    3. python src/pipelines/fiinratings_pdf_extractor.py

EXPECTED OUTPUT:
    - Logs showing PDF processing progress
    - CSV file at data/processed/fiinratings_extracted.csv
"""

import os
import json
import logging
import pdfplumber
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai

# Try importing paddleocr, handle gracefully if not installed
try:
    from paddleocr import PaddleOCR
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

# Load environment variables from .env file (looks in CWD and parent dirs)
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. DEFINE SCHEMA FOR STRUCTURED EXTRACTION
# ==========================================
class FinancialData(BaseModel):
    company_name: Optional[str] = Field(default=None, description="Tên công ty được xếp hạng")
    ticker: Optional[str] = Field(default=None, description="Mã chứng khoán (nếu có)")
    rating_date: Optional[str] = Field(default=None, description="Ngày xếp hạng (định dạng YYYY-MM-DD)")
    rating_detail: Optional[str] = Field(default=None, description="Bậc xếp hạng tín dụng doanh nghiệp (VD: AAA, AA+, A-, BBB, ...)")
    sector: Optional[str] = Field(default=None, description="Ngành nghề hoạt động")
    
    # Financial Indicators (12 variables matching merged_credit_rating_common_3groups.csv)
    current_ratio: Optional[float] = Field(default=None, description="Tỷ số thanh toán hiện hành (Current Ratio)")
    debt_equity_ratio: Optional[float] = Field(default=None, description="Hệ số Nợ trên Vốn chủ sở hữu (Debt/Equity Ratio)")
    gross_profit_margin: Optional[float] = Field(default=None, description="Biên lợi nhuận gộp (Gross Profit Margin)")
    operating_profit_margin: Optional[float] = Field(default=None, description="Biên lợi nhuận hoạt động (Operating Profit Margin)")
    ebit_margin: Optional[float] = Field(default=None, description="Biên EBIT (EBIT Margin)")
    pretax_profit_margin: Optional[float] = Field(default=None, description="Biên lợi nhuận trước thuế (Pretax Profit Margin)")
    net_profit_margin: Optional[float] = Field(default=None, description="Biên lợi nhuận ròng (Net Profit Margin)")
    asset_turnover: Optional[float] = Field(default=None, description="Vòng quay tổng tài sản (Asset Turnover)")
    roe: Optional[float] = Field(default=None, description="Tỷ suất lợi nhuận trên vốn chủ sở hữu (ROE)")
    roa: Optional[float] = Field(default=None, description="Tỷ suất lợi nhuận trên tổng tài sản (ROA)")
    operating_cashflow_ps: Optional[float] = Field(default=None, description="Dòng tiền hoạt động trên mỗi cổ phiếu (Operating Cashflow per Share)")
    free_cashflow_ps: Optional[float] = Field(default=None, description="Dòng tiền tự do trên mỗi cổ phiếu (Free Cashflow per Share)")

# ==========================================
# 2. EXTRACTION PIPELINE
# ==========================================
class PDFExtractorPipeline:
    def __init__(self, api_key: str, use_ocr_fallback: bool = True):
        """Initialize the PDF extraction pipeline.
        
        Args:
            api_key: Google Gemini API key.
            use_ocr_fallback: Whether to use PaddleOCR for scanned PDFs.
        """
        if not api_key or api_key == "DUMMY_KEY_FOR_TESTING":
            raise ValueError(
                "GEMINI_API_KEY chưa được cấu hình. "
                "Hãy đặt key trong file .env hoặc biến môi trường."
            )
        
        # Initialize the new google-genai Client
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash"
        logger.info(f"Initialized Gemini client with model: {self.model_name}")
        
        # Initialize PaddleOCR
        self.use_ocr_fallback = use_ocr_fallback and HAS_PADDLE
        if self.use_ocr_fallback:
            logger.info("Initializing PaddleOCR...")
            self.ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False)

    def extract_text_from_native_pdf(self, pdf_path: str) -> str:
        """Extract text and basic table structures using pdfplumber (Fast & 100% accurate for Native PDF)."""
        full_text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # 1. Extract raw text
                    text = page.extract_text(x_tolerance=1, y_tolerance=1)
                    if text:
                        full_text += text + "\n"
                    
                    # 2. Extract tables (to preserve row/column relationships for financial data)
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            cleaned_row = [str(cell).replace('\n', ' ') for cell in row if cell]
                            full_text += " | ".join(cleaned_row) + "\n"
        except Exception as e:
            logger.error(f"pdfplumber error on {pdf_path}: {e}")
            
        return full_text

    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """Fallback OCR for scanned parts using PaddleOCR."""
        logger.info(f"Using PaddleOCR fallback for {pdf_path}...")
        # Note: For real OCR on PDFs, you need pdf2image to convert PDF pages to PIL images first.
        # This is a placeholder for the logic.
        ocr_text = ""
        # import pdf2image
        # images = pdf2image.convert_from_path(pdf_path)
        # for img in images:
        #     result = self.ocr.ocr(np.array(img), cls=True)
        #     for line in result[0]:
        #         ocr_text += line[1][0] + "\n"
        return ocr_text

    def parse_with_llm(self, text: str) -> Optional[Dict[str, Any]]:
        """Pass the extracted text/tables to Gemini to map to the FinancialData schema.
        
        Uses google-genai SDK v2 structured output with Pydantic schema.
        """
        prompt = f"""
        Bạn là một chuyên gia phân tích dữ liệu tài chính. 
        Dưới đây là nội dung thô được trích xuất từ một báo cáo xếp hạng tín dụng doanh nghiệp (có thể chứa văn bản và bảng biểu).
        
        Nhiệm vụ của bạn là:
        1. Tìm các thông tin cơ bản: Tên công ty, Mã CK (nếu có), Ngày xếp hạng, Bậc xếp hạng (VD: AA+, BBB-, ...), Ngành nghề.
        2. Tìm 12 chỉ số tài chính (có thể nằm trong đoạn văn hoặc các bảng Tóm tắt tài chính).
        
        QUY TẮC:
        - Nếu chỉ số là % (VD: 15.5%), hãy chuyển thành số thập phân (0.155).
        - Nếu không tìm thấy thông tin nào, bắt buộc trả về null (không được bịa số).
        - Hiểu các từ đồng nghĩa (VD: "Biên LN gộp" = "Gross Profit Margin").
        
        VĂN BẢN TRÍCH XUẤT:
        ====================
        {text[:60000]}
        ====================
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": FinancialData,
                },
            )
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"LLM Parsing error: {e}")
            return None

    def process_document(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        filename = os.path.basename(pdf_path)
        logger.info(f"Processing: {filename}")
        
        # Bước 1: Trích xuất Native Text
        text = self.extract_text_from_native_pdf(pdf_path)
        native_len = len(text.strip())
        logger.info(f"  [pdfplumber] Extracted {native_len} chars from {filename}")
        
        # Bước 2: Dự phòng OCR (nếu text quá ngắn, tức là file scan hoặc chứa ảnh)
        if native_len < 500 and self.use_ocr_fallback:
            logger.info(f"  [OCR] Text too short ({native_len} chars), trying PaddleOCR fallback...")
            ocr_text = self.extract_text_with_ocr(pdf_path)
            text += "\n" + ocr_text
            logger.info(f"  [OCR] Added {len(ocr_text.strip())} chars from OCR")
        elif native_len < 500:
            logger.warning(f"  [SKIP] {filename}: scanned PDF ({native_len} chars) and no OCR available")
            return None
            
        # Bước 3: Đẩy qua LLM để cấu trúc hóa (Structured Output)
        final_len = len(text.strip())
        if final_len > 50:
            logger.info(f"  [LLM] Sending {final_len} chars to Gemini for extraction...")
            result = self.parse_with_llm(text)
            if result:
                result['source_file'] = filename
                logger.info(f"  [OK] Extracted rating={result.get('rating_detail')} for {result.get('company_name')}")
                return result
            else:
                logger.warning(f"  [FAIL] LLM returned no data for {filename}")
        else:
            logger.warning(f"  [SKIP] {filename}: insufficient text ({final_len} chars)")
                
        return None

# ==========================================
# 3. RUNNER LOGIC
# ==========================================
if __name__ == "__main__":
    import sys
    import time
    sys.stdout.reconfigure(encoding='utf-8')
    
    API_KEY = os.environ.get("GEMINI_API_KEY", "")
    PDF_DIR = Path("data/external/fiinratings_output/pdfs")
    
    if not API_KEY:
        logger.error(
            "GEMINI_API_KEY not found. "
            "Please set it in src/pipelines/.env or as an environment variable."
        )
        raise SystemExit(1)
    
    pipeline = PDFExtractorPipeline(api_key=API_KEY, use_ocr_fallback=True)
    
    results: list[Dict[str, Any]] = []
    skipped: list[str] = []
    failed: list[str] = []
    
    # Process all PDFs
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    total = len(pdf_files)
    logger.info(f"Found {total} PDF files in {PDF_DIR}")
    
    # For testing, limit to first N files. Set to None to run all.
    LIMIT: Optional[int] = None # Change to None to process all files
    batch = pdf_files[:LIMIT] if LIMIT else pdf_files
    logger.info(f"Processing {len(batch)}/{total} files (LIMIT={LIMIT})")
    
    for i, pdf_file in enumerate(batch, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(batch)}] {pdf_file.name}")
        logger.info(f"{'='*60}")
        try:
            data = pipeline.process_document(str(pdf_file))
            if data:
                results.append(data)
            else:
                skipped.append(pdf_file.name)
        except Exception as e:
            logger.error(f"UNEXPECTED ERROR on {pdf_file.name}: {e}")
            failed.append(pdf_file.name)
        
        # Rate limiting: avoid hitting Gemini API quota
        if i < len(batch):
            time.sleep(1)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("EXTRACTION SUMMARY")
    logger.info(f"  Total processed: {len(batch)}")
    logger.info(f"  Successful: {len(results)}")
    logger.info(f"  Skipped (no text/data): {len(skipped)}")
    logger.info(f"  Failed (errors): {len(failed)}")
    if failed:
        logger.info(f"  Failed files: {failed}")
    logger.info(f"{'='*60}")
            
    # Export to CSV matching the format of merged_credit_rating_common_3groups.csv
    if results:
        df = pd.DataFrame(results)
        
        # Reorder columns to match the target CSV roughly
        cols = ['source_file', 'rating_detail', 'company_name', 'ticker', 'rating_date', 'sector',
                'current_ratio', 'debt_equity_ratio', 'gross_profit_margin', 
                'operating_profit_margin', 'ebit_margin', 'pretax_profit_margin', 
                'net_profit_margin', 'asset_turnover', 'roe', 'roa', 
                'operating_cashflow_ps', 'free_cashflow_ps']
        df = df[[c for c in cols if c in df.columns]]
        
        output_path = Path("data/processed/fiinratings_extracted.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Extraction complete! Saved to {output_path}")
        print(df.to_string())
    else:
        logger.warning("No data extracted from any PDF files.")
