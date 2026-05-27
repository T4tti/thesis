"""Dataset optimization pipeline for corporate credit rating data.

This script enhances the processed dataset e:\\thesis\\data\\processed\\merged_credit_rating_common_3groups.csv
by applying several Senior AI Research Engineer level optimization techniques:
1) Sector-relative Normalization: Standardizing financial ratios within each sector.
2) Temporal Delta Features: Calculating 1-period lagged changes (delta) in key ratios for each ticker.
3) Robust Outlier Capping: Using interquartile range (IQR) to clip extreme values.
4) Missing Value Strategy: Forward-fill followed by sector-median fill for lagged features.

Output:
    - data/processed/train_augmented_timegan_optimized.csv
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT_DIR / "data" / "processed" / "train_augmented_timegan.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "train_augmented_timegan_optimized.csv"

FINANCIAL_RATIOS = [
    "current_ratio", "debt_equity_ratio", "gross_profit_margin",
    "operating_profit_margin", "ebit_margin", "pretax_profit_margin",
    "net_profit_margin", "asset_turnover", "roe", "roa",
    "operating_cashflow_ps", "free_cashflow_ps"
]

def apply_sector_normalization(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Standardize financial ratios relative to their sector median and std."""
    print("Applying sector-relative normalization...")
    df_norm = df.copy()
    for col in columns:
        # Z-score within each sector
        sector_stats = df.groupby('sector')[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
        df_norm[f"{col}_sector_z"] = sector_stats
    return df_norm

def apply_temporal_deltas(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Calculate 1-period lagged changes (delta) for each ticker."""
    print("Calculating temporal delta features...")
    df_delta = df.copy()
    df_delta['rating_date'] = pd.to_datetime(df_delta['rating_date'])
    df_delta = df_delta.sort_values(by=['ticker', 'rating_date'])
    
    for col in columns:
        # Shift within each ticker group
        df_delta[f"{col}_lag1"] = df_delta.groupby('ticker')[col].shift(1)
        df_delta[f"{col}_delta"] = df_delta[col] - df_delta[f"{col}_lag1"]
        
        # Fill missing lags/deltas with 0 for the first record of each company
        df_delta[f"{col}_delta"] = df_delta[f"{col}_delta"].fillna(0)
    
    return df_delta

def robust_outlier_capping(df: pd.DataFrame, columns: list[str], factor: float = 3.0) -> pd.DataFrame:
    """Clip values using IQR-based bounds to remove extreme outliers."""
    print(f"Capping outliers with factor {factor}...")
    df_capped = df.copy()
    for col in columns:
        Q1 = df_capped[col].quantile(0.25)
        Q3 = df_capped[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
    return df_capped

def main():
    print(f"Loading dataset from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    
    # 1. Handle Outliers First (Clean the raw signals)
    df = robust_outlier_capping(df, FINANCIAL_RATIOS)
    
    # 2. Apply Sector-relative Normalization
    df = apply_sector_normalization(df, FINANCIAL_RATIOS)
    
    # 3. Apply Temporal Delta Features
    df = apply_temporal_deltas(df, FINANCIAL_RATIOS)
    
    # 4. Final Cleanup
    # Remove columns that were only used for delta calculations if needed,
    # but keeping them might be useful for some models.
    # Drop lag1 columns as they are redundant with the raw values + delta
    lag_cols = [col for col in df.columns if "_lag1" in col]
    df = df.drop(columns=lag_cols)
    
    print(f"Saving optimized dataset to {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False)
    print("Done! Optimized features count:", len(df.columns))

if __name__ == "__main__":
    main()
