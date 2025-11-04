"""Module for grading submissions against the reference solution."""
import pandas as pd
from typing import Tuple

def grade_solution(submission_df: pd.DataFrame, reference_df: pd.DataFrame) -> Tuple[float, str]:
    """Grade the submission against reference solution with stricter criteria."""
    if submission_df is None:
        return 0.0, "No DataFrame returned"
        
    # Print both DataFrames for comparison
    print("\nReference DataFrame (Original):")
    print("="*50)
    print(reference_df)
    
    print("\nSubmission DataFrame (After Pipeline):")
    print("="*50)
    print(submission_df)
    print("\n")
        
    score = 0.0
    messages = []
    
    # Check 1: Basic DataFrame validity and shape
    if not all(col in submission_df.columns for col in ['age', 'height', 'weight']):
        return 0.0, "Missing required columns"
    
    # Check 2: Age filtering (0.2 points)
    age_valid = True
    if submission_df['age'].max() <= 100:
        if not submission_df['age'].isna().any():
            score += 0.2
            messages.append("✓ Age filtering correct")
        else:
            age_valid = False
            messages.append("✗ NaN values in age column")
    else:
        age_valid = False
        messages.append("✗ Ages > 100 present")
    
    # Check 3: Missing value imputation (0.2 points)
    if not submission_df.isna().any().any():
        # Verify means are close to reference solution
        height_mean_diff = abs(submission_df['height'].mean() - reference_df.loc[reference_df['age'] <= 100, 'height'].mean())
        weight_mean_diff = abs(submission_df['weight'].mean() - reference_df.loc[reference_df['age'] <= 100, 'weight'].mean())
        
        if height_mean_diff < 1.0 and weight_mean_diff < 1.0:
            score += 0.2
            messages.append("✓ Missing values imputed correctly")
        else:
            messages.append("✗ Imputed values deviate from expected means")
    else:
        messages.append("✗ Missing values remain")
    
    # Check 4: Standardization (0.3 points)
    tolerance = 0.005  # Stricter tolerance
    standardization_score = 0.0
    
    for col in ['age', 'height', 'weight']:
        mean = submission_df[col].mean()
        std = submission_df[col].std()
        if abs(mean) < tolerance and abs(1 - std) < tolerance:
            standardization_score += 0.1
            messages.append(f"✓ {col} standardized correctly")
        else:
            messages.append(f"✗ {col}: mean={mean:.4f}, std={std:.4f}")
    
    score += standardization_score
    
    # Check 5: Data integrity (0.3 points)
    try:
        # Check row count is appropriate after filtering
        expected_rows = len(reference_df[reference_df['age'] <= 100])
        if len(submission_df) == expected_rows:
            score += 0.15
            messages.append("✓ Correct number of rows after filtering")
        else:
            messages.append("✗ Incorrect number of rows")
        
        # Check data hasn't been unnecessarily modified
        if age_valid and all(submission_df.index == range(len(submission_df))):
            score += 0.15
            messages.append("✓ Index properly reset")
        else:
            messages.append("✗ Index issues or data integrity problems")
    except:
        messages.append("✗ Error checking data integrity")
    
    return score, "\n".join(messages)