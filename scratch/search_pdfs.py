import os
import re
from pypdf import PdfReader

def extract_and_search_pdf(pdf_path, keywords):
    """Đọc PDF và tìm kiếm từ khóa kèm theo context"""
    results = {}
    try:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue
            
            for kw in keywords:
                # Case insensitive find all
                for match in re.finditer(kw, text, re.IGNORECASE):
                    idx = match.start()
                    start = max(0, idx - 100)
                    end = min(len(text), idx + 100)
                    context = text[start:end].replace("\n", " ")
                    
                    if kw not in results:
                        results[kw] = []
                    results[kw].append((page_num + 1, context))
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return results

def main():
    pdfs = [
        r"e:\thesis\LeNguyenHoangPhu_bcao.pdf",
        r"e:\thesis\KL_word nghiên cứu - Copy.pdf"
    ]
    keywords = ["Gompertz", "Choquet", "fuzzy", "phương pháp tổng hợp", "kịch bản", "FR-"]
    report_path = r"e:\thesis\scratch\search_pdfs_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        for pdf in pdfs:
            if not os.path.exists(pdf):
                f.write(f"File {pdf} does not exist.\n")
                continue
            f.write(f"\nSearching in PDF: {pdf}\n")
            f.write("="*60 + "\n")
            
            results = extract_and_search_pdf(pdf, keywords)
            for kw, matches in results.items():
                f.write(f"  Found '{kw}': {len(matches)} occurrences\n")
                # Write first 5 occurrences
                for i, (page, context) in enumerate(matches[:10]):
                    f.write(f"    Page {page}: ... {context} ...\n")
                    
    print(f"Search report written to {report_path}")

if __name__ == "__main__":
    main()
