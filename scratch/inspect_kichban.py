import pandas as pd
import openpyxl
import sys

def main():
    file_path = r"e:\thesis\kichban.xlsx"
    report_path = r"e:\thesis\scratch\kichban_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Loading {file_path}...\n")
        
        # Load Excel file
        xl = pd.ExcelFile(file_path)
        f.write(f"Sheets in excel file: {xl.sheet_names}\n")
        
        for sheet_name in xl.sheet_names:
            f.write("\n" + "="*50 + "\n")
            f.write(f"SHEET: {sheet_name}\n")
            f.write("="*50 + "\n")
            df = xl.parse(sheet_name)
            f.write(f"Shape: {df.shape}\n")
            f.write("Columns:\n")
            f.write(str(df.columns.tolist()) + "\n")
            f.write("\nData:\n")
            f.write(df.to_string() + "\n")
            
    print(f"Report written to {report_path}")

if __name__ == "__main__":
    main()
