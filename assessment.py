import os
import google.generativeai as genai
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
import json

# Configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY environment variable")
genai.configure(api_key=GEMINI_API_KEY)

# Assessment prompt
PROMPT = """
Given a DataFrame with columns 'age', 'height', and 'weight', write a Python function that:

1. REMOVES invalid rows:
   - Where age > 100 OR
   - Where age is NaN/missing

2. FILLS missing values:
   - Calculate means for 'height' and 'weight' AFTER age filtering
   - Replace NaN values with respective column means

3. STANDARDIZES all columns:
   - Transform each column to have mean=0 and std=1
   - Use formula: (value - mean) / std
   - Handle zero standard deviation cases

Requirements:
- Use pandas
- Make a copy of input DataFrame
- Use proper indexing (avoid chained assignments)
- Return the cleaned DataFrame
"""

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

def grade_solution(submission_df: pd.DataFrame, reference_df: pd.DataFrame) -> Tuple[float, str]:
    """Grade the submission against reference solution."""
    if submission_df is None:
        return 0.0, "No DataFrame returned"
        
    score = 0.0
    messages = []
    
    # Check 1: Basic DataFrame validity
    if not all(col in submission_df.columns for col in ['age', 'height', 'weight']):
        return 0.0, "Missing required columns"
    
    # Check 2: Age filtering (0.3 points)
    if (submission_df['age'].max() <= 100 and 
        not submission_df['age'].isna().any()):
        score += 0.3
        messages.append("✓ Age filtering correct")
    else:
        messages.append("✗ Invalid ages present")
        
    # Check 3: Missing value imputation (0.3 points)
    if not submission_df.isna().any().any():
        score += 0.3
        messages.append("✓ No missing values")
    else:
        messages.append("✗ Missing values remain")
        
    # Check 4: Standardization (0.4 points)
    tolerance = 0.01
    standardization_score = 0.0
    
    for col in ['age', 'height', 'weight']:
        mean = submission_df[col].mean()
        std = submission_df[col].std()
        if abs(mean) < tolerance and abs(std - 1) < tolerance:
            standardization_score += 0.4/3
            messages.append(f"✓ {col} standardized correctly")
        else:
            messages.append(f"✗ {col}: mean={mean:.4f}, std={std:.4f}")
            
    score += standardization_score
    
    return score, "\n".join(messages)

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

def extract_code_from_response(response_text: str) -> str:
    """Extract Python code from LLM response."""
    import re
    
    # Try to find code between ```python ... ```
    python_match = re.search(r'```python\n(.*?)\n```', response_text, re.DOTALL)
    if python_match:
        return python_match.group(1).strip()
    
    # Try to find code between ``` ... ```
    code_match = re.search(r'```\n?(.*?)\n?```', response_text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # If no code blocks found, return cleaned text
    return response_text.strip()

def save_assessment_results(results: Dict, filename: str = "assessment_results.json"):
    """Save assessment results to file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

def run_assessment(num_trials: int = 10, should_save: bool = True) -> Dict:
    """Run the full assessment multiple times."""
    results = {
        'scores': [],
        'feedback': [],
        'trials': []
    }
    
    for trial in range(num_trials):
        print(f"\n{'='*60}")
        print(f"Trial {trial + 1}/{num_trials}")
        print('='*60)
        
        # Generate test data
        test_df = generate_test_data(seed=42 + trial)
        
        try:
            # Get model's solution
            model = genai.GenerativeModel('gemini-2.5-pro')
            response = model.generate_content(
                f"{PROMPT}\n\nHere's the DataFrame to clean:\n{test_df.to_string()}\n\n"
                "Return only the Python code that implements the solution."
            )
            
            # Extract and execute code
            code = extract_code_from_response(response.text)
            print("\nGenerated Code:")
            print("-" * 40)
            print(code)
            print("-" * 40)
            
            # Create namespace for execution
            namespace = {'pd': pd, 'np': np, 'df': test_df.copy()}
            exec(code, namespace)
            
            # Try to get result
            result_df = None
            if 'clean_and_standardize_dataframe' in namespace:
                result_df = namespace['clean_and_standardize_dataframe'](test_df.copy())
            elif 'clean_dataframe' in namespace:
                result_df = namespace['clean_dataframe'](test_df.copy())
            elif 'df' in namespace:
                result_df = namespace['df']
            
            if result_df is None:
                raise ValueError("No DataFrame returned")
            
            # Grade solution
            score, feedback = grade_solution(result_df, test_df)
            
            results['scores'].append(score)
            results['feedback'].append(feedback)
            results['trials'].append({
                'trial': trial + 1,
                'score': score,
                'feedback': feedback,
                'code': code
            })
            
            print(f"\nScore: {score:.2f}")
            print("Feedback:")
            print(feedback)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            results['scores'].append(0.0)
            results['feedback'].append(f"Error: {str(e)}")
            results['trials'].append({
                'trial': trial + 1,
                'score': 0.0,
                'feedback': f"Error: {str(e)}",
                'code': code if 'code' in locals() else None
            })
    
    # Calculate statistics
    scores = results['scores']
    pass_threshold = 0.8
    pass_rate = sum(1 for s in scores if s >= pass_threshold) / len(scores)
    avg_score = sum(scores) / len(scores)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Number of trials: {num_trials}")
    print(f"Pass rate (>= {pass_threshold:.1f}): {pass_rate:.1%}")
    print(f"Average score: {avg_score:.2f}")
    print(f"Score range: {min(scores):.2f} - {max(scores):.2f}")
    
    results['summary'] = {
        'num_trials': num_trials,
        'pass_rate': pass_rate,
        'avg_score': avg_score,
        'min_score': min(scores),
        'max_score': max(scores)
    }
    
    if should_save:
        save_assessment_results(results)
    
    return results

if __name__ == "__main__":
    # Run assessment with 10 trials
    results = run_assessment(num_trials=5)