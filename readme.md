## Task: Clean and Standardize a Dataset

**Objective:**  
Teach the model to clean tabular data using Pandas.

**Prompt Summary:**  
Remove invalid ages, impute missing height/weight with mean, and standardize all columns.

**Pass Criteria:**  
- Rows with age > 100 or NaN removed  
- No missing values remain  
- Columns have mean≈0, std≈1  

**Expected pass rate:** ~20–35%

**Skills tested:**  
- Data filtering  
- Imputation  
- Normalization  
- Following multi-step instructions

# Score Breakdown
0.0  # Failed (execution error or missing columns)
0.3  # Only correct filtering (removed age > 100 and NaN)
0.4  # Correct filtering + successful imputation
0.7  # Above + attempted standardization but not perfect
1.0  # Perfect solution (all steps correct)