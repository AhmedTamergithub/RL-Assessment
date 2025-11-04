"""Module containing the reference/golden solution for data cleaning task."""
import pandas as pd

def reference_solution(df: pd.DataFrame) -> pd.DataFrame:
    """Reference implementation of the data cleaning task."""
    # Make a copy to avoid modifying original
    cleaned_df = df.copy()
    
    # 1. Remove invalid ages
    valid_age_mask = (cleaned_df['age'].notna()) & (cleaned_df['age'] <= 100)
    cleaned_df = cleaned_df[valid_age_mask].reset_index(drop=True)
    
    # 2. Fill missing values
    for col in ['height', 'weight']:
        mean_val = cleaned_df[col].mean()
        cleaned_df.loc[cleaned_df[col].isna(), col] = mean_val
    
    # 3. Standardize columns
    for col in ['age', 'height', 'weight']:
        mean = cleaned_df[col].mean()
        std = cleaned_df[col].std()
        if std != 0:
            cleaned_df[col] = (cleaned_df[col] - mean) / std
    
    return cleaned_df