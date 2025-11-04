import google.generativeai as genai
import pandas as pd
import numpy as np
import os
from prompt import PROMPT
from tool import generate_dataset
from grader import grade

# Configure Gemini - make sure to set your API key in environment
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY environment variable")

genai.configure(api_key=GEMINI_API_KEY)

def extract_code_from_response(response_text: str) -> str:
    """Extract Python code from LLM response."""
    if "```python" in response_text:
        code = response_text.split("```python")[1].split("```")[0]
    elif "```" in response_text:
        code = response_text.split("```")[1].split("```")[0]
    else:
        code = response_text
    return code.strip()

def create_cleaning_function(code: str):
    """Create a function from the LLM's code."""
    # Create the function that will be called by the grader
    namespace = {}
    try:
        exec(code, namespace)
        
        def cleaning_function(df):
            # Create a copy of namespace to avoid state between runs
            local_namespace = {'df': df.copy(), 'pd': pd, 'np': np}
            exec(code, local_namespace)
            # Return either 'df' or 'result' if defined
            return local_namespace.get('df', local_namespace.get('result'))
            
        return cleaning_function
    except Exception as e:
        print(f"Error creating function: {e}")
        return None

def run_single_test(test_number: int = 1) -> tuple[float, str]:
    """Run a single test with the LLM."""
    print(f"\n{'='*60}")
    print(f"Test #{test_number}")
    print(f"{'='*60}")
    
    # Generate test data
    df = generate_dataset(seed=42 + test_number)
    
    # Create the context for the LLM
    context = f"""
{PROMPT}

Here is the DataFrame you need to clean:
```python
df = pd.DataFrame({df.to_dict()})
```

Return only the Python code that cleans this DataFrame according to the requirements.
Make sure your code uses pandas and follows all the steps in order.
"""
    
    try:
        # Query Gemini
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(context)
        
        # Extract and print the code
        code = extract_code_from_response(response.text)
        print("\nGenerated Code:")
        print("-" * 40)
        print(code)
        print("-" * 40)
        
        # Create cleaning function
        cleaning_function = create_cleaning_function(code)
        if cleaning_function is None:
            return 0.0, "Failed to create function from code"
        
        # Grade the solution
        score, message = grade(cleaning_function)
        print(f"\nScore: {score:.2f}")
        print(f"Message: {message}")
        
        return score, message
        
    except Exception as e:
        print(f"Error during test: {e}")
        return 0.0, str(e)

def main(num_tests: int = 10):
    """Run multiple tests and calculate statistics."""
    scores = []
    messages = []
    
    for i in range(num_tests):
        score, message = run_single_test(i + 1)
        scores.append(score)
        messages.append(message)
    
    # Calculate statistics
    pass_threshold = 0.8
    pass_rate = sum(1 for s in scores if s >= pass_threshold) / len(scores)
    avg_score = sum(scores) / len(scores)
    
    # Print results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Number of tests: {num_tests}")
    print(f"Pass rate (>= {pass_threshold:.1f}): {pass_rate:.1%}")
    print(f"Average score: {avg_score:.2f}")
    print(f"Score range: {min(scores):.2f} - {max(scores):.2f}")
    print("\nAll scores:")
    for i, (score, msg) in enumerate(zip(scores, messages), 1):
        print(f"Test {i}: {score:.2f} - {msg}")

if __name__ == "__main__":
    main(num_tests=10)