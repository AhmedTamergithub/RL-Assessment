"""Module for generating test data for the assessment."""
import numpy as np
import pandas as pd

def generate_test_data(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic test data with various edge cases."""
    np.random.seed(seed)
    n_rows = 10
    
    # Generate base data
    df = pd.DataFrame({
        'age': np.random.normal(45, 20, n_rows),  # Some will be > 100
        'height': np.random.normal(170, 15, n_rows),
        'weight': np.random.normal(70, 15, n_rows)
    })
    
    # Add invalid ages (>100)
    invalid_indices = np.random.choice(n_rows, 5, replace=False)
    df.loc[invalid_indices[:3], 'age'] = np.random.uniform(101, 120, 3)
    
    # Add NaN values
    df.loc[invalid_indices[3:], 'age'] = np.nan
    df.loc[np.random.choice(n_rows, 3), 'height'] = np.nan
    df.loc[np.random.choice(n_rows, 3), 'weight'] = np.nan
    
    return df