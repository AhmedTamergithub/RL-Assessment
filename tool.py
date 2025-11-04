import pandas as pd
import numpy as np

def generate_dataset(seed=42):
    """Generate a synthetic dataset with missing values and outliers."""
    np.random.seed(seed)
    df = pd.DataFrame({
        "age": np.random.randint(10, 120, 20).astype(float),
        "height": np.random.normal(170, 10, 20),
        "weight": np.random.normal(70, 15, 20)
    })

    # Introduce missing values and outliers
    df.loc[[2, 7], "age"] = np.nan
    df.loc[[4, 9], "height"] = np.nan
    df.loc[[5, 10], "weight"] = np.nan
    return df
