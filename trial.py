
import pandas as pd
import numpy as np
def clean_and_standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes the DataFrame.
    """
    # Make a copy at the start to avoid any chained assignment issues
    df_cleaned = df.copy()
    
    # Step 1: Remove rows with invalid ages (> 100 or NaN)
    # More explicit filtering to ensure we catch all invalid cases
    valid_age_mask = (df_cleaned['age'].notna()) & (df_cleaned['age'] <= 100)
    df_cleaned = df_cleaned[valid_age_mask].reset_index(drop=True)
    
    # Step 2: Fill missing values
    # Use loc to avoid chained assignment warning
    for col in ['height', 'weight']:
        mean_val = df_cleaned[col].mean()
        df_cleaned.loc[df_cleaned[col].isna(), col] = mean_val
    
    # Step 3: Standardize columns
    for col in ['age', 'height', 'weight']:
        mean = df_cleaned[col].mean()
        std = df_cleaned[col].std()
        if std != 0:  # Prevent division by zero
            df_cleaned[col] = (df_cleaned[col] - mean) / std
        else:
            df_cleaned[col] = 0
    
    return df_cleaned



def test_solution():
    """
    Test the DataFrame cleaning function with various test cases.
    Prints detailed information about the transformations.
    """
    # Create test data with more edge cases
    test_df = pd.DataFrame({
        'age': [25, np.nan, 101, 50, 75, 120, 30],
        'height': [170, 180, np.nan, 165, 175, 168, np.nan],
        'weight': [70, np.nan, 80, 65, 75, 72, 68]
    })
    
    print("\n=== Data Cleaning Test Results ===")
    print("\n1. Original DataFrame:")
    print(test_df)
    print("\nOriginal Statistics:")
    print(f"Shape: {test_df.shape}")
    print(f"Missing values:\n{test_df.isna().sum()}")
    print(f"Age range: {test_df['age'].min()} - {test_df['age'].max()}")
    
    # Clean the data
    result = clean_and_standardize_dataframe(test_df)
    
    print("\n2. Cleaned DataFrame:")
    print(result)
    print("\nCleaned Statistics:")
    print(f"Shape: {result.shape}")
    print(f"Missing values:\n{result.isna().sum()}")
    print(f"Age range: {result['age'].min()} - {result['age'].max()}")
    
    # Verify results with detailed feedback
    print("\n3. Verification Steps:")
    
    # Check row removal
    print(f"✓ Rows removed: {len(test_df) - len(result)} (Expected: rows with age > 100 or NaN)")
    assert len(result) < len(test_df), "Should have removed invalid ages"
    
    # Check age limits
    print(f"✓ Maximum age: {result['age'].max():.1f} (Expected: ≤ 100)")
    assert result['age'].max() <= 100, "Should have no ages > 100"
    
    # Check missing values
    print(f"✓ Missing values: {result.isna().sum().sum()} (Expected: 0)")
    assert result.isna().sum().sum() == 0, "Should have no NaN values"
    
    # Check standardization
    print("\n4. Standardization Results:")
    for col in result.columns:
        mean = result[col].mean()
        std = result[col].std()
        print(f"{col}:")
        print(f"  Mean: {mean:.6f} (Expected: ~0)")
        print(f"  Std:  {std:.6f} (Expected: ~1)")
        assert abs(mean) < 0.001, f"{col} mean should be ~0"
        assert abs(std - 1) < 0.001, f"{col} std should be ~1"
    
    print("\n✅ All tests passed successfully!")
    
if __name__ == "__main__":
    test_solution()