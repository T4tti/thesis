"""
Merge và Xử lý 2 Dataset Corporate Credit Rating
Theo quy trình từ merge_dataset.md

Input:
  - corporate_rating.csv (File 1)
  - corporateCreditRatingWithFinancialRatios.csv (File 2)

Output:
  - merged_credit_rating_full.csv    (Phương án C - Full: giữ mọi cột)
  - merged_credit_rating_common.csv  (Phương án C - Common: chỉ cột chung)
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Chia an toàn, tránh chia cho 0 gây inf."""
    return numerator / denominator.replace(0, np.nan)


def safe_group_zscore(series: pd.Series) -> pd.Series:
    """Z-score theo group, giữ NaN và trả về 0 khi std = 0."""
    std = series.std(ddof=0)
    if pd.isna(std) or np.isclose(std, 0):
        return pd.Series(np.where(series.notna(), 0.0, np.nan), index=series.index)
    return (series - series.mean()) / std

# ============================================================
# BƯỚC 1: Load dữ liệu
# ============================================================
print("=" * 60)
print("BƯỚC 1: Load dữ liệu")
print("=" * 60)

df1 = pd.read_csv(RAW_DIR / "corporate_rating.csv")
df2 = pd.read_csv(RAW_DIR / "corporateCreditRatingWithFinancialRatios.csv")

print(f"File 1 shape: {df1.shape}")
print(f"File 2 shape: {df2.shape}")
print(f"File 1 missing: {df1.isnull().sum().sum()}")
print(f"File 2 missing: {df2.isnull().sum().sum()}")


# ============================================================
# BƯỚC 2: Chuẩn hóa tên cột (Column Mapping)
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 2: Chuẩn hóa tên cột")
print("=" * 60)

# Rename File 1 columns
df1 = df1.rename(columns={
    "Rating":                               "rating_detail",
    "Name":                                 "company_name",
    "Symbol":                               "ticker",
    "Rating Agency Name":                   "rating_agency",
    "Date":                                 "rating_date",
    "Sector":                               "sector",
    "currentRatio":                         "current_ratio",
    "quickRatio":                           "quick_ratio",
    "cashRatio":                            "cash_ratio",
    "daysOfSalesOutstanding":               "days_sales_outstanding",
    "netProfitMargin":                      "net_profit_margin",
    "pretaxProfitMargin":                   "pretax_profit_margin",
    "grossProfitMargin":                    "gross_profit_margin",
    "operatingProfitMargin":                "operating_profit_margin",
    "returnOnAssets":                       "roa",
    "returnOnCapitalEmployed":              "return_on_capital_employed",
    "returnOnEquity":                       "roe",
    "assetTurnover":                        "asset_turnover",
    "fixedAssetTurnover":                   "fixed_asset_turnover",
    "debtEquityRatio":                      "debt_equity_ratio",
    "debtRatio":                            "debt_ratio",
    "effectiveTaxRate":                     "effective_tax_rate",
    "freeCashFlowOperatingCashFlowRatio":   "fcf_ocf_ratio",
    "freeCashFlowPerShare":                 "free_cashflow_ps",
    "cashPerShare":                         "cash_per_share",
    "companyEquityMultiplier":              "equity_multiplier",
    "ebitPerRevenue":                       "ebit_margin",
    "enterpriseValueMultiple":              "enterprise_value_multiple",
    "operatingCashFlowPerShare":            "operating_cashflow_ps",
    "operatingCashFlowSalesRatio":          "ocf_sales_ratio",
    "payablesTurnover":                     "payables_turnover",
})
df1["source"] = "corporate_rating"

# Rename File 2 columns
df2 = df2.rename(columns={
    "Rating Agency":                        "rating_agency",
    "Corporation":                          "company_name",
    "Rating":                               "rating_detail",
    "Rating Date":                          "rating_date",
    "CIK":                                  "cik",
    "Binary Rating":                        "binary_rating",
    "SIC Code":                             "sic_code",
    "Sector":                               "sector",
    "Ticker":                               "ticker",
    "Current Ratio":                        "current_ratio",
    "Long-term Debt / Capital":             "long_term_debt_capital",
    "Debt/Equity Ratio":                    "debt_equity_ratio",
    "Gross Margin":                         "gross_profit_margin",
    "Operating Margin":                     "operating_profit_margin",
    "EBIT Margin":                          "ebit_margin",
    "EBITDA Margin":                        "ebitda_margin",
    "Pre-Tax Profit Margin":                "pretax_profit_margin",
    "Net Profit Margin":                    "net_profit_margin",
    "Asset Turnover":                       "asset_turnover",
    "ROE - Return On Equity":               "roe",
    "Return On Tangible Equity":            "return_on_tangible_equity",
    "ROA - Return On Assets":               "roa",
    "ROI - Return On Investment":           "roi",
    "Operating Cash Flow Per Share":        "operating_cashflow_ps",
    "Free Cash Flow Per Share":             "free_cashflow_ps",
})
df2["source"] = "credit_rating_financial"

print(f"File 1 columns after rename: {df1.columns.tolist()}")
print(f"File 2 columns after rename: {df2.columns.tolist()}")


# ============================================================
# BƯỚC 3: Chuẩn hóa cột Rating
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 3: Chuẩn hóa cột Rating")
print("=" * 60)

RATING_CLASSES = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]

def map_to_class(rating: str) -> str:
    """Map detailed rating (với modifier +/-) về nhóm lớn."""
    rating = str(rating).strip()
    for base in RATING_CLASSES:
        if rating.startswith(base):
            return base
    return rating  # trả về giá trị gốc nếu không match

df1["rating_class"] = df1["rating_detail"].apply(map_to_class)
df2["rating_class"] = df2["rating_detail"].apply(map_to_class)

print("File 1 - rating_class distribution:")
print(df1["rating_class"].value_counts().sort_index())
print("\nFile 2 - rating_class distribution:")
print(df2["rating_class"].value_counts().sort_index())


# ============================================================
# BƯỚC 4: Chuẩn hóa cột Sector
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 4: Chuẩn hóa cột Sector")
print("=" * 60)

# Mapping từ abbreviation (File 2) sang tên chuẩn
# Note: File 2 chỉ dùng abbreviation, KHÔNG có SIC code số trong cột Sector
SECTOR_MAP_F2 = {
    "BusEq":  "Technology",
    "Chems":  "Basic Industries",
    "Durbl":  "Consumer Durables",
    "Enrgy":  "Energy",
    "Hlth":   "Health Care",
    "Manuf":  "Capital Goods",
    "Money":  "Finance",
    "NoDur":  "Consumer Non-Durables",
    "Other":  "Miscellaneous",
    "Shops":  "Consumer Services",
    "Telcm":  "Public Utilities",
    "Utils":  "Public Utilities",
}

df2["sector"] = df2["sector"].map(SECTOR_MAP_F2).fillna(df2["sector"])

# File 1: Chuẩn hóa tên sector (giữ nguyên "Transportation")
# Đảm bảo consistent casing / naming
SECTOR_MAP_F1 = {
    "Consumer Durables":        "Consumer Durables",
    "Energy":                   "Energy",
    "Basic Industries":         "Basic Industries",
    "Consumer Services":        "Consumer Services",
    "Technology":               "Technology",
    "Capital Goods":            "Capital Goods",
    "Public Utilities":         "Public Utilities",
    "Health Care":              "Health Care",
    "Consumer Non-Durables":    "Consumer Non-Durables",
    "Transportation":           "Transportation",
    "Miscellaneous":            "Miscellaneous",
    "Finance":                  "Finance",
}
df1["sector"] = df1["sector"].map(SECTOR_MAP_F1).fillna(df1["sector"])

print("File 1 - sector after mapping:")
print(df1["sector"].value_counts())
print("\nFile 2 - sector after mapping:")
print(df2["sector"].value_counts())

unmapped_f2 = df2["sector"].isnull().sum()
if unmapped_f2 > 0:
    print(f"\nWARNING: {unmapped_f2} rows in File 2 have unmapped sectors!")


# ============================================================
# BƯỚC 5: Chuẩn hóa đơn vị Financial Ratios
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 5: Chuẩn hóa đơn vị (decimal vs percentage)")
print("=" * 60)

# File 1: Margins/returns ở dạng decimal (0-1)
# File 2: Margins/returns ở dạng percentage (0-100) → chia 100

pct_cols_file2 = [
    "gross_profit_margin",
    "operating_profit_margin",
    "ebit_margin",
    "ebitda_margin",
    "pretax_profit_margin",
    "net_profit_margin",
    "roe",
    "return_on_tangible_equity",
    "roa",
    "roi",
    # NOTE: long_term_debt_capital đã ở dạng decimal (0-1) trong File 2, KHÔNG cần chia 100
    # Ví dụ: American States Water = 0.4551 (tức 45.51%), ADP = 0.0072 (tức 0.72%)
]

# Chỉ convert các cột tồn tại trong df2
pct_cols_file2 = [c for c in pct_cols_file2 if c in df2.columns]

print(f"Columns to convert (÷100) in File 2: {pct_cols_file2}")
print("\nFile 2 stats TRƯỚC khi convert:")
print(df2[pct_cols_file2].describe().round(4))

for col in pct_cols_file2:
    df2[col] = df2[col] / 100.0

print("\nFile 2 stats SAU khi convert:")
print(df2[pct_cols_file2].describe().round(4))

# Kiểm tra alignment giữa 2 files cho các cột chung
common_check_cols = ["current_ratio", "gross_profit_margin", "net_profit_margin", "roa"]
print("\nKiểm tra alignment đơn vị sau chuẩn hóa:")
for col in common_check_cols:
    if col in df1.columns and col in df2.columns:
        f1_median = df1[col].median()
        f2_median = df2[col].median()
        print(f"  {col}: File1 median={f1_median:.4f}, File2 median={f2_median:.4f}")


# ============================================================
# BƯỚC 6: Chuẩn hóa cột Date
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 6: Chuẩn hóa cột Date")
print("=" * 60)

df1["rating_date"] = pd.to_datetime(df1["rating_date"], format="%m/%d/%Y", errors="coerce")
df2["rating_date"] = pd.to_datetime(df2["rating_date"], format="%m/%d/%Y", errors="coerce")

print(f"File 1 - date range: {df1['rating_date'].min()} → {df1['rating_date'].max()}")
print(f"File 2 - date range: {df2['rating_date'].min()} → {df2['rating_date'].max()}")

date_null_f1 = df1["rating_date"].isnull().sum()
date_null_f2 = df2["rating_date"].isnull().sum()
if date_null_f1 or date_null_f2:
    print(f"WARNING: Date parse failures - File1: {date_null_f1}, File2: {date_null_f2}")


# ============================================================
# BƯỚC 7: Gộp 2 DataFrame
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 7: Gộp 2 DataFrame")
print("=" * 60)

# Uppercase ticker để đồng bộ
df1["ticker"] = df1["ticker"].str.upper().str.strip()
df2["ticker"] = df2["ticker"].str.upper().str.strip()

df_merged = pd.concat([df1, df2], axis=0, ignore_index=True)
print(f"Merged shape: {df_merged.shape}")
print(f"Columns: {df_merged.columns.tolist()}")


# ============================================================
# BƯỚC 8: Xử lý bản ghi trùng lặp
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 8: Xử lý duplicate")
print("=" * 60)

KEY_COLS = ["ticker", "rating_agency", "rating_date"]
n_before = len(df_merged)

duplicates = df_merged[df_merged.duplicated(subset=KEY_COLS, keep=False)].copy()
print(f"Số bản ghi trùng key (ticker + agency + date): {len(duplicates)}")
print(f"Số cặp duplicate: {len(duplicates) // 2}")

if len(duplicates) > 0:
    print("\nSample duplicates:")
    print(duplicates[KEY_COLS + ["rating_detail", "source"]].head(10))

# Chiến lược: ưu tiên bản từ File 2 (rating chi tiết hơn) khi rating_date giống nhau
# Đồng thời, giữ bản có nhiều non-null values hơn nếu rating như nhau
df_merged["_non_null_count"] = df_merged.notna().sum(axis=1)
df_merged["_source_priority"] = df_merged["source"].map({
    "credit_rating_financial": 1,   # ưu tiên cao hơn
    "corporate_rating": 2,
})

df_merged = df_merged.sort_values(
    by=["ticker", "rating_agency", "rating_date", "_source_priority", "_non_null_count"],
    ascending=[True, True, True, True, False]
)
df_merged = df_merged.drop_duplicates(subset=KEY_COLS, keep="first")

# Cleanup columns
df_merged = df_merged.drop(columns=["_non_null_count", "_source_priority"])

n_after = len(df_merged)
print(f"\nRows removed as duplicates: {n_before - n_after}")
print(f"Rows after dedup: {n_after}")


# ============================================================
# BƯỚC 9: Tái tạo binary_rating cho toàn bộ dataset
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 9: Tái tạo binary_rating")
print("=" * 60)

INVESTMENT_GRADES = ["AAA", "AA", "A", "BBB"]
df_merged["binary_rating"] = df_merged["rating_class"].apply(
    lambda x: 1 if x in INVESTMENT_GRADES else 0
)

print("Binary rating distribution:")
print(df_merged["binary_rating"].value_counts())
print(f"  Investment Grade (1): {df_merged['binary_rating'].eq(1).sum()} ({df_merged['binary_rating'].eq(1).mean()*100:.1f}%)")
print(f"  Speculative Grade (0): {df_merged['binary_rating'].eq(0).sum()} ({df_merged['binary_rating'].eq(0).mean()*100:.1f}%)")


# ============================================================
# BƯỚC 10: Xử lý Missing Data - Phương án C (2 versions)
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 10: Phân tích Missing Data")
print("=" * 60)

# Cột chung (shared) giữa 2 files
SHARED_FINANCIAL_COLS = [
    "current_ratio",
    "debt_equity_ratio",
    "gross_profit_margin",
    "operating_profit_margin",
    "ebit_margin",
    "pretax_profit_margin",
    "net_profit_margin",
    "asset_turnover",
    "roe",
    "roa",
    "operating_cashflow_ps",
    "free_cashflow_ps",
]

# Metadata cột chung
SHARED_META_COLS = [
    "rating_detail", "rating_class", "binary_rating",
    "company_name", "ticker", "rating_agency", "rating_date",
    "sector", "source",
]

# Missing analysis
missing_pct = (df_merged.isnull().sum() / len(df_merged) * 100).round(1)
missing_report = missing_pct[missing_pct > 0].sort_values(ascending=False)
print("Cột có missing values (%):")
print(missing_report.to_string())


# ============================================================
# BƯỚC 11: Xử lý outliers - Winsorize 1%-99%
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 11: Winsorize outliers (1%-99%)")
print("=" * 60)

ALL_FINANCIAL_COLS = list(set(SHARED_FINANCIAL_COLS) | {
    # File 1 only
    "quick_ratio", "cash_ratio", "days_sales_outstanding",
    "return_on_capital_employed", "fixed_asset_turnover", "debt_ratio",
    "effective_tax_rate", "fcf_ocf_ratio", "cash_per_share",
    "equity_multiplier", "enterprise_value_multiple", "ocf_sales_ratio",
    "payables_turnover",
    # File 2 only
    "long_term_debt_capital", "ebitda_margin",
    "return_on_tangible_equity", "roi",
})

# Chỉ winsorize các cột tồn tại trong merged df
cols_to_winsorize = [c for c in ALL_FINANCIAL_COLS if c in df_merged.columns]

winsorize_stats = {}
for col in cols_to_winsorize:
    series = df_merged[col].dropna()
    if len(series) == 0:
        continue
    p1 = series.quantile(0.01)
    p99 = series.quantile(0.99)
    n_clipped = ((df_merged[col] < p1) | (df_merged[col] > p99)).sum()
    df_merged[col] = df_merged[col].clip(lower=p1, upper=p99)
    winsorize_stats[col] = {"p1": p1, "p99": p99, "n_clipped": n_clipped}

print(f"Winsorized {len(cols_to_winsorize)} columns.")
print("\nColumns with most clipping:")
clip_df = pd.DataFrame(winsorize_stats).T
clip_df = clip_df.sort_values("n_clipped", ascending=False).head(10)
print(clip_df.to_string())


# ============================================================
# BƯỚC 12: Feature Engineering chuẩn Moody's/S&P (proxy)
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 12: Feature Engineering chuẩn Moody's/S&P")
print("=" * 60)

# Trụ cột rating phổ biến: leverage, liquidity, profitability, cashflow, peer-relative
df_merged["rating_year"] = df_merged["rating_date"].dt.year

df_merged["fe_leverage_liquidity"] = safe_divide(
    df_merged["debt_equity_ratio"],
    df_merged["current_ratio"],
)

df_merged["fe_cashflow_conversion"] = safe_divide(
    df_merged["free_cashflow_ps"],
    df_merged["operating_cashflow_ps"],
)
df_merged["fe_cashflow_spread_ps"] = (
    df_merged["operating_cashflow_ps"] - df_merged["free_cashflow_ps"]
)

profitability_cols = [
    "gross_profit_margin",
    "operating_profit_margin",
    "ebit_margin",
    "pretax_profit_margin",
    "net_profit_margin",
    "roe",
    "roa",
]
profitability_cols = [c for c in profitability_cols if c in df_merged.columns]

if profitability_cols:
    df_merged["fe_profitability_composite"] = df_merged[profitability_cols].mean(axis=1, skipna=True)
else:
    df_merged["fe_profitability_composite"] = np.nan

df_merged["fe_earnings_quality_gap"] = (
    df_merged["operating_profit_margin"] - df_merged["net_profit_margin"]
)

peer_base_cols = [
    "current_ratio",
    "debt_equity_ratio",
    "ebit_margin",
    "net_profit_margin",
    "roa",
    "roe",
    "operating_cashflow_ps",
    "free_cashflow_ps",
]
peer_base_cols = [c for c in peer_base_cols if c in df_merged.columns]

engineered_peer_cols = []
for col in peer_base_cols:
    sector_col = f"fe_{col}_sector_z"
    sector_year_col = f"fe_{col}_sector_year_z"

    df_merged[sector_col] = df_merged.groupby("sector")[col].transform(safe_group_zscore)
    df_merged[sector_year_col] = df_merged.groupby(["sector", "rating_year"])[col].transform(safe_group_zscore)
    engineered_peer_cols.extend([sector_col, sector_year_col])

SHARED_ENGINEERED_COLS = [
    "fe_leverage_liquidity",
    "fe_cashflow_conversion",
    "fe_cashflow_spread_ps",
    "fe_profitability_composite",
    "fe_earnings_quality_gap",
] + engineered_peer_cols

# Dọn inf và winsorize engineered features để giảm ảnh hưởng outlier
df_merged[SHARED_ENGINEERED_COLS] = df_merged[SHARED_ENGINEERED_COLS].replace([np.inf, -np.inf], np.nan)

for col in SHARED_ENGINEERED_COLS:
    series = df_merged[col].dropna()
    if len(series) == 0:
        continue
    p1 = series.quantile(0.01)
    p99 = series.quantile(0.99)
    df_merged[col] = df_merged[col].clip(lower=p1, upper=p99)

print(f"Đã tạo {len(SHARED_ENGINEERED_COLS)} engineered features.")
print("Sample engineered columns:")
print(SHARED_ENGINEERED_COLS[:10])


# ============================================================
# BƯỚC 13: Kiểm tra Hidden Missing (zeros)
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 13: Kiểm tra Hidden Missing (zeros)")
print("=" * 60)

print("Cột có giá trị 0 đáng ngờ:")
for col in SHARED_FINANCIAL_COLS:
    if col in df_merged.columns:
        zero_count = (df_merged[col] == 0).sum()
        total_valid = df_merged[col].notna().sum()
        if zero_count > 0 and total_valid > 0:
            zero_pct = zero_count / total_valid * 100
            print(f"  {col}: {zero_count} zeros ({zero_pct:.1f}% of valid)")


# ============================================================
# BƯỚC 14: Export Kết quả (Phương án C)
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 14: Export kết quả")
print("=" * 60)

# Version Full: toàn bộ cột
FULL_COLS = SHARED_META_COLS + [c for c in df_merged.columns
                                 if c not in SHARED_META_COLS and c != "binary_rating"]
# Sắp xếp cột hợp lý
col_order_full = (
    SHARED_META_COLS
    + [c for c in df_merged.columns if c not in SHARED_META_COLS]
)
col_order_full = [c for c in col_order_full if c in df_merged.columns]
df_full = df_merged[col_order_full].copy()

# Version Common: chỉ cột chung
col_order_common = SHARED_META_COLS + SHARED_FINANCIAL_COLS + SHARED_ENGINEERED_COLS
col_order_common = [c for c in col_order_common if c in df_merged.columns]
df_common = df_merged[col_order_common].dropna(subset=SHARED_FINANCIAL_COLS, how="all").copy()

# Export
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def export_with_fallback(df: pd.DataFrame, target_path: Path) -> Path:
    """Ghi CSV, nếu file đang bị lock thì ghi sang file fallback."""
    try:
        df.to_csv(target_path, index=False)
        return target_path
    except PermissionError:
        fallback_path = target_path.with_name(f"{target_path.stem}_new{target_path.suffix}")
        df.to_csv(fallback_path, index=False)
        print(f"WARNING: File đang bị khóa, đã ghi fallback: {fallback_path}")
        return fallback_path


full_output_path = export_with_fallback(df_full, PROCESSED_DIR / "merged_credit_rating_full.csv")
common_output_path = export_with_fallback(df_common, PROCESSED_DIR / "merged_credit_rating_common.csv")

print(f"\nFull dataset  : {df_full.shape}  → {full_output_path}")
print(f"Common dataset: {df_common.shape} → {common_output_path}")

# Summary
print("\n" + "=" * 60)
print("TỔNG KẾT")
print("=" * 60)
print(f"File 1 (gốc): {df1.shape[0]:,} rows")
print(f"File 2 (gốc): {df2.shape[0]:,} rows")
print(f"Sau merge   : {len(df_merged_raw := df_merged):,} rows")
print(f"Full export : {df_full.shape[0]:,} rows × {df_full.shape[1]} cols")
print(f"Common export: {df_common.shape[0]:,} rows × {df_common.shape[1]} cols")
print(f"\nRating class distribution (Full):")
print(df_full["rating_class"].value_counts().sort_index())
print(f"\nSector distribution (Full):")
print(df_full["sector"].value_counts())
print(f"\nSource distribution:")
print(df_full["source"].value_counts())
print("\nDone!")
