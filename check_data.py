#!/usr/bin/env python3
"""
Quick data status check for AMR Intelligence Platform
"""

import pandas as pd
import os
from pathlib import Path

def check_data_status():
    print("🔬 AMR Intelligence Platform - Data Status Check")
    print("=" * 55)

    # Check processed data
    processed_file = Path("data/processed/amr_panel_data.csv")
    if processed_file.exists():
        try:
            df = pd.read_csv(processed_file)
            print(f"✅ Processed panel data loaded: {len(df)} records")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Countries: {df['Country'].nunique()} unique")
            print(f"   Time range: {df['Year'].min()} - {df['Year'].max()}")

            # Check data types
            print("\nData availability:")
            real_data_cols = []
            missing_cols = []

            for col in df.columns:
                if col in ['Country', 'Year']: continue
                non_null_count = df[col].notna().sum()
                total_count = len(df)
                pct = (non_null_count / total_count) * 100
                if pct > 80:
                    real_data_cols.append(col)
                elif non_null_count < 10:
                    missing_cols.append(col)

            if real_data_cols:
                print(f"   📊 Real data columns ({len(real_data_cols)}): {real_data_cols}")
            if missing_cols:
                print(f"   ⚠️  Mostly missing columns ({len(missing_cols)}): {missing_cols}")

            print("\nSample data:")
            print(df.head(3).to_string())

        except Exception as e:
            print(f"❌ Error reading processed data: {e}")

    # Check raw data files
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        raw_files = list(raw_dir.glob("*.json")) + list(raw_dir.glob("*.csv"))
        print(f"\n📁 Raw data sources ({len(raw_files)} files):")
        for file in raw_files:
            print(f"   • {file.name}")

    # Check results directory
    results_dir = Path("results")
    if results_dir.exists():
        result_files = list(results_dir.glob("*"))
        print(f"\n📊 Results ({len(result_files)} files):")
        for file in result_files:
            print(f"   • {file.name}")

    print("\n🎯 Recommendations:")
    if processed_file.exists():
        df = pd.read_csv(processed_file)
        if len(df) > 10 and df['Country'].nunique() > 3:
            print("   ✅ Ready for analysis - suficiente data available")
            print("   📊 Run: python main.py analysis")
            print("   📈 For forecasting: python main.py forecast")
        else:
            print("   ⚠️  Limited data - consider manual downloads")
    else:
        print("   ❓ No processed data found - run pipeline first")

if __name__ == "__main__":
    check_data_status()
