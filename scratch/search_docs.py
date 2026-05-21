import zipfile
import xml.etree.ElementTree as ET
import re
import os

def extract_text_from_docx(docx_path):
    """Trích xuất text thô từ file .docx (zip archive)"""
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            # Find all text elements
            texts = []
            for elem in root.iter():
                if elem.tag.endswith('t'): # w:t
                    if elem.text:
                        texts.append(elem.text)
            return " ".join(texts)
    except Exception as e:
        return f"Error reading {docx_path}: {e}"

def main():
    docs = [
        r"e:\thesis\Thesis.docx",
        r"e:\thesis\Lý thuyết đề tài.docx",
        r"e:\thesis\KL_word nghiên cứu - Copy.docx"
    ]
    
    keywords = ["Gompertz", "Choquet", "fuzzy", "phương pháp tổng hợp", "kịch bản", "FR-"]
    report_path = r"e:\thesis\scratch\search_docs_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        for doc in docs:
            if not os.path.exists(doc):
                f.write(f"File {doc} does not exist.\n")
                continue
            f.write(f"\nSearching in: {doc}\n")
            text = extract_text_from_docx(doc)
            f.write(f"Total character length: {len(text)}\n")
            
            # Search for keyword matches with context
            for kw in keywords:
                # Case insensitive search
                matches = [m.start() for m in re.finditer(kw, text, re.IGNORECASE)]
                if matches:
                    f.write(f"  Found '{kw}': {len(matches)} matches\n")
                    # Print first 5 matches context
                    for i, idx in enumerate(matches[:5]):
                        start = max(0, idx - 100)
                        end = min(len(text), idx + 100)
                        context = text[start:end].replace("\n", " ")
                        f.write(f"    Match {i+1}: ... {context} ...\n")
                else:
                    f.write(f"  Keyword '{kw}' not found.\n")
                    
    print(f"Search report written to {report_path}")

if __name__ == "__main__":
    main()
