import pandas as pd
import numpy as np

def main():
    # Paths (relative to repository root)
    raw_path = 'data/Malawi_rbm_data.csv'
    out_path = 'data/cleaned_inflation_prepped.csv'
    # Load raw data
    raw_df = pd.read_csv(raw_path)
    # Parse dates
    raw_df['date'] = pd.to_datetime(raw_df['date'], format='%Y-%m-%d')
    raw_df = raw_df.sort_values('date')
    # Pivot to wide format
    wide_df = raw_df.pivot_table(index='date', columns='indicator', values='value')
    # Detect duplicate rows (date, indicator)
    dup_count = raw_df.duplicated(subset=['date', 'indicator']).sum()
    print(f'Duplicate date‑indicator rows: {dup_count}')
    # Ensure complete monthly index
    full_index = pd.date_range(start=wide_df.index.min(), end=wide_df.index.max(), freq='MS')
    wide_df = wide_df.reindex(full_index)
    wide_df.index.name = 'date'
    print('Shape after reindexing (including NaNs for missing months):', wide_df.shape)
    # Report missing values per indicator
    missing_counts = wide_df.isna().sum()
    print('Missing observations per indicator:')
    print(missing_counts)
    # CPI rebasing flag – mark large month‑to‑month changes (>15 %)
    cpi = wide_df['consumer_price_index']
    cpi_change = cpi.pct_change().abs()
    cpi_rebase_flag = (cpi_change > 0.15).astype(int)
    wide_df['cpi_rebase_flag'] = cpi_rebase_flag
    rebase_years = wide_df.loc[wide_df['cpi_rebase_flag']==1].index.year.unique()
    print('CPI rebasing flagged in years:', list(rebase_years))
    # Save cleaned dataset
    wide_df.to_csv(out_path, float_format='%.4f')
    print('Saved cleaned dataset to', out_path)

if __name__ == '__main__':
    main()
