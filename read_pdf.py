# read_pdf.py
import argparse
from pypdf import PdfReader
import sys

def read_pdf_content(pdf_path: str) -> str:
    """
    Reads the text content from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text content from the PDF.
    """
    try:
        reader = PdfReader(pdf_path)
        text_content = ""
        for page in reader.pages:
            text_content += page.extract_text() + "\n"
        return text_content
    except Exception as e:
        return f"Error reading PDF file: {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from a PDF file.")
    parser.add_argument("pdf_file", type=str, help="The path to the PDF file.")
    parser.add_argument("-o", "--output", type=str, help="Output file to save the extracted text. If not provided, prints to console.")
    args = parser.parse_args()

    content = read_pdf_content(args.pdf_file)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Content successfully written to {args.output}")
        except Exception as e:
            print(f"Error writing to output file: {e}", file=sys.stderr)
    else:
        # Fallback for printing to console, but with explicit encoding if possible
        # This might still fail depending on the console's capabilities,
        # so writing to a file is the more robust solution.
        try:
            print(content)
        except UnicodeEncodeError:
            print("UnicodeEncodeError: Could not print to console. Please use the -o option to save to a file.", file=sys.stderr)
