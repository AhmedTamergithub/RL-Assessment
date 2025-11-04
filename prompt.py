PROMPT = """
You are given a CSV file containing three columns: age, height, and weight.
Write Python code using pandas to:

1. Remove rows where 'age' > 100 or 'age' is NaN.
2. Fill missing 'height' or 'weight' values with their respective column means.
3. Standardize each column ('age', 'height', 'weight') so each has mean 0 and std 1.

Return the cleaned DataFrame as output.
"""
