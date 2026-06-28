"""
FiinRatings Credit Rating Scraper
==================================
Thu thập toàn bộ kết quả xếp hạng tín dụng doanh nghiệp từ fiinratings.vn
- Lưu danh sách vào CSV
- Tải xuống báo cáo PDF (tiếng Việt & tiếng Anh)

Cấu trúc HTML thực tế:
- AJAX trả về các thẻ <tr> không có <table> bao ngoài
- col[0]: Ngày công bố
- col[1]: Tên tổ chức (link)
- col[2]: Ngành
- col[3]: Loại xếp hạng (StatusRating)
- col[4]: Loại ý kiến
- col[5]: Mức xếp hạng (có <span class=icon-info> cần loại bỏ)
- col[6]: Triển vọng (trong <span class=type-text>)
- col[7]: ISIN (trong <span>, có thể là "---")
- col[8]: Báo cáo PDF (JSON trong div lồng nhau)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import logging
import re
from pathlib import Path
from datetime import datetime
from urllib.parse import unquote

# ─── Cấu hình ───────────────────────────────────────────────────────────────
BASE_URL    = "https://fiinratings.vn"
AJAX_URL    = f"{BASE_URL}/vi/ratings/indexajax"
TOTAL_PAGES = 15
DELAY_PAGE  = 1.5
DELAY_PDF   = 0.8
TIMEOUT     = 45

ROOT_DIR    = Path(__file__).resolve().parents[2]
OUTPUT_DIR  = ROOT_DIR / "data" / "external" / "fiinratings_output"
PDF_DIR     = OUTPUT_DIR / "pdfs"
CSV_FILE    = OUTPUT_DIR / "credit_ratings.csv"
LOG_FILE    = OUTPUT_DIR / "scraper.log"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html, */*; q=0.01",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": f"{BASE_URL}/vi/ket-qua-xep-hang/ket-qua-xep-hang-tin-nhiem.html",
}


# ─── Logging ─────────────────────────────────────────────────────────────────
def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

logger = logging.getLogger(__name__)


# ─── Parse một hàng <tr> ────────────────────────────────────────────────────
def parse_row(tr) -> dict | None:
    """Parse một thẻ <tr> thành dict. Trả về None nếu không phải hàng dữ liệu."""
    cols = tr.find_all("td")
    if len(cols) < 8:
        return None

    # col[0]: Ngày công bố
    date = cols[0].get_text(strip=True)
    if not date or not re.match(r"\d{2}/\d{2}/\d{4}", date):
        return None  # Bỏ qua hàng không phải dữ liệu (footer/pagination)

    # col[1]: Tên tổ chức + URL
    link = cols[1].find("a")
    company = link.get_text(strip=True) if link else cols[1].get_text(strip=True)
    detail_url = ""
    if link and link.get("href"):
        href = link["href"]
        detail_url = href if href.startswith("http") else BASE_URL + href

    # col[2]: Ngành
    sector = cols[2].get_text(strip=True)

    # col[3]: Loại xếp hạng
    rating_type = cols[3].get_text(strip=True)

    # col[4]: Loại ý kiến
    rating_class = cols[4].get_text(strip=True)

    # col[5]: Mức xếp hạng (bỏ span.icon-info, chỉ lấy text node đầu tiên)
    rating_span = cols[5].find("span", class_="icon-info")
    if rating_span:
        rating_span.decompose()
    rating_grade = cols[5].get_text(strip=True)

    # col[6]: Triển vọng
    outlook_span = cols[6].find("span", class_="type-text")
    outlook = outlook_span.get_text(strip=True) if outlook_span else cols[6].get_text(strip=True)

    # col[7]: ISIN
    isin_span = cols[7].find("span")
    isin_text = isin_span.get_text(strip=True) if isin_span else cols[7].get_text(strip=True)
    isin = "" if isin_text in ("---", "-", "") else isin_text

    # col[8]: PDF links - lấy JSON từ div thứ HAI trong div[style=display:none]
    vi_pdf = ""
    en_pdf = ""
    hidden_div = cols[8].find("div", style=lambda s: s and "display:none" in s.replace(" ", ""))
    if hidden_div:
        sub_divs = hidden_div.find_all("div", recursive=False)
        # div đầu thường là null, div thứ hai có link thực
        for sub in sub_divs:
            text = sub.get_text(strip=True)
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    vi_url = data.get("vi") or ""
                    en_url = data.get("en") or ""
                    if vi_url or en_url:
                        vi_pdf = vi_url.replace("//UploadFile", "/UploadFile")
                        en_pdf = en_url.replace("//UploadFile", "/UploadFile")
                        break
            except (json.JSONDecodeError, TypeError):
                continue

    return {
        "Ngày":          date,
        "Tên tổ chức":   company,
        "URL trang":     detail_url,
        "Ngành":         sector,
        "Loại xếp hạng": rating_type,
        "Phân loại":     rating_class,
        "Mức xếp hạng":  rating_grade,
        "Triển vọng":    outlook,
        "ISIN":          isin,
        "PDF_VI":        vi_pdf,
        "PDF_EN":        en_pdf,
    }


# ─── Parse một trang HTML ────────────────────────────────────────────────────
def parse_page(html: str) -> list[dict]:
    """Parse HTML từ AJAX trả về (các <tr> không có <table>)."""
    # Wrap trong <table> để BeautifulSoup parse đúng
    soup = BeautifulSoup(f"<table><tbody>{html}</tbody></table>", "html.parser")
    rows = soup.find_all("tr")
    records = []
    for tr in rows:
        rec = parse_row(tr)
        if rec:
            records.append(rec)
    return records


# ─── Thu thập toàn bộ trang ──────────────────────────────────────────────────
def scrape_all_pages(session: requests.Session) -> list[dict]:
    all_records = []
    for page in range(1, TOTAL_PAGES + 1):
        logger.info(f"Trang {page}/{TOTAL_PAGES}...")
        try:
            resp = session.get(AJAX_URL, params={"page": page},
                               headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            records = parse_page(resp.text)
            all_records.extend(records)
            logger.info(f"  -> {len(records)} bản ghi | Tổng: {len(all_records)}")
        except requests.RequestException as e:
            logger.error(f"  Lỗi trang {page}: {e}")
        if page < TOTAL_PAGES:
            time.sleep(DELAY_PAGE)
    return all_records


# ─── Tải PDF ─────────────────────────────────────────────────────────────────
def safe_filename(name: str, max_len: int = 80) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    return re.sub(r'\s+', "_", name.strip())[:max_len]


def download_pdf(session: requests.Session, url: str, dest: Path) -> bool:
    if not url:
        return False
    if dest.exists() and dest.stat().st_size > 1024:
        logger.info(f"    Đã có: {dest.name}")
        return True
    try:
        resp = session.get(url, headers={"User-Agent": HEADERS["User-Agent"]},
                           timeout=TIMEOUT, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        size_kb = dest.stat().st_size // 1024
        logger.info(f"    OK ({size_kb} KB): {dest.name}")
        return True
    except Exception as e:
        logger.error(f"    Lỗi: {e} | URL: {url}")
        if dest.exists():
            dest.unlink()
        return False


def download_all_pdfs(session: requests.Session, records: list[dict]):
    ok = err = skip = 0
    total = len(records)

    for i, rec in enumerate(records, 1):
        company = safe_filename(rec["Tên tổ chức"])
        date    = rec["Ngày"].replace("/", "-")

        for lang, url in [("VI", rec["PDF_VI"])]:
            if not url:
                skip += 1
                continue

            # Dùng tên file gốc từ URL
            orig = unquote(url.split("/")[-1])
            dest = PDF_DIR / (orig if orig.lower().endswith(".pdf") else f"{date}_{company}_{lang}.pdf")

            logger.info(f"[{i}/{total}] {lang}: {dest.name}")
            if download_pdf(session, url, dest):
                ok += 1
            else:
                err += 1
            time.sleep(DELAY_PDF)

    return ok, err, skip


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    setup_logging()
    t0 = datetime.now()
    logger.info("=" * 60)
    logger.info("FiinRatings Scraper khởi động")
    logger.info(f"Thời gian: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    session = requests.Session()

    # BƯỚC 1: Thu thập danh sách
    logger.info("\n[BƯỚC 1] Thu thập danh sách xếp hạng...")
    records = scrape_all_pages(session)

    if not records:
        logger.error("Không thu thập được dữ liệu nào. Dừng.")
        return

    # BƯỚC 2: Lưu CSV
    logger.info(f"\n[BƯỚC 2] Lưu {len(records)} bản ghi -> {CSV_FILE}")
    df = pd.DataFrame(records)
    df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
    logger.info("  -> Đã lưu CSV")

    # Thống kê
    logger.info(f"\n  Tổng bản ghi: {len(df)}")
    for col in ["Ngành", "Mức xếp hạng", "Triển vọng"]:
        if col in df.columns:
            logger.info(f"  [{col}] {df[col].value_counts().to_dict()}")

    # BƯỚC 3: Tải PDF
    logger.info(f"\n[BƯỚC 3] Tải PDF vào {PDF_DIR}...")
    ok, err, skip = download_all_pdfs(session, records)

    # Tổng kết
    elapsed = datetime.now() - t0
    logger.info("\n" + "=" * 60)
    logger.info("HOÀN THÀNH")
    logger.info(f"  Thời gian chạy : {elapsed}")
    logger.info(f"  Bản ghi CSV    : {len(records)}")
    logger.info(f"  PDF tải OK     : {ok}")
    logger.info(f"  PDF lỗi        : {err}")
    logger.info(f"  PDF bỏ qua     : {skip}  (không có link)")
    logger.info(f"  Output dir     : {OUTPUT_DIR.resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
