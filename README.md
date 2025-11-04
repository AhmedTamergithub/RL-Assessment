# LLM Data Cleaning Assessment System

## 🎯 The Challenge: Automated Data Cleaning

We're testing how well Large Language Models (specifically Google's Gemini) can handle a fundamental data science task: **cleaning and standardizing messy data**. 

### The Problem Given to the LLM

Given a pandas DataFrame with common data quality issues:
- Invalid age values (> 100 years)
- Missing data (NaN values)
- Non-standardized numerical distributions

The LLM must generate code that:
1. Filters out problematic rows
2. Imputes missing values appropriately
3. Standardizes the numerical distributions

This tests the model's ability to:
- Understand data cleaning requirements
- Generate correct pandas code
- Handle edge cases
- Follow best practices

## 🏗️ System Architecture

The assessment system is divided into four main modules:

1. **Data Generator** (`data_generator.py`)
   - Creates synthetic test data
   - Injects realistic data quality issues
   - Ensures consistent test conditions

2. **Golden Model** (`golden_model.py`)
   - Implements the reference solution
   - Defines the expected behavior
   - Serves as the ground truth

3. **Grader** (`grader.py`)
   - Evaluates LLM solutions
   - Implements strict scoring criteria
   - Provides detailed feedback

4. **Assessment Pipeline** (`assessment.py`)
   - Manages LLM interaction
   - Executes generated code safely
   - Collects performance metrics

## 📊 Evaluation Criteria

Solutions are graded on a 1.0 scale across four dimensions:

1. **Age Filtering** (0.2 points)
   - Remove rows with age > 100
   - Remove rows with NaN ages

2. **Missing Value Imputation** (0.2 points)
   - Calculate means after filtering
   - Correctly fill NaN values

3. **Standardization** (0.3 points)
   - Transform to mean=0, std=1
   - Handle all columns correctly

4. **Data Integrity** (0.3 points)
   - Maintain correct row count
   - Proper index management
   - Data consistency

## 🔄 Workflow

1. System generates test data with controlled issues
2. Prompts Gemini with the cleaning task
3. Extracts and executes the generated code
4. Grades the solution against reference implementation
5. Provides detailed feedback and scores
6. Repeats across multiple trials for reliability

## 📈 Performance Expectations

- Expected pass rate: 20-40%
- Passing score threshold: ≥ 0.5
- Common failure points:
  - Improper handling of NaN values
  - Incorrect standardization
  - Index management issues
  - Data integrity violations

## 🔧 Quick Start Guide

1. **Setup**:
```bash
export GEMINI_API_KEY='your-key-here'
```

2. **Run Assessment**:
```bash
# Run with default 10 trials
python assessment.py

# Or specify number of trials
python assessment.py --num_trials 5
```

3. **Results**:
- Saves to `assessment_results.json`
- Shows per-trial feedback
- Displays summary statistics

## 📊 Key Metrics

- **Score Range**: 0.0 - 1.0
- **Pass Threshold**: ≥ 0.9
- **Tracked Metrics**:
  - Average score
  - Pass rate
  - Min/max scores
  - Per-component performance

## � Sample Output

Here's an example of what the system produces during assessment:

### Input Data
```
Reference DataFrame (Original):
==================================================
          age      height     weight
0   39.189937  162.415238  82.341299
1   47.242561         NaN  83.740357
2  104.213162  150.903509  95.752722
3   17.782201  181.201746        NaN
4  117.014620         NaN  97.430741
5   44.040178  168.337977  86.236184
6         NaN  152.189950  75.529364
7         NaN  154.876611        NaN
8   37.979646  159.652074        NaN
9  109.693890         NaN  54.348691
```

### Model's Solution Output
```
Submission DataFrame (After Pipeline):
==================================================
        age        height        weight
0  0.168920 -6.614117e-01 -1.264863e+00
1  0.868983  3.426298e-15 -2.620470e-01
3 -1.692185  1.603342e+00 -1.018604e-14
5  0.590581  5.258701e-02  1.526910e+00
8  0.063702 -9.945171e-01 -1.018604e-14
```

### Grading Results
```
Score: 0.65
Feedback:
✓ Age filtering correct
✗ Imputed values deviate from expected means
✓ age standardized correctly
✓ height standardized correctly
✓ weight standardized correctly
✓ Correct number of rows after filtering
✗ Index issues or data integrity problems
```

This example demonstrates:
1. **Input Data Issues**:
   - Invalid ages (>100): rows 2, 4, 9
   - Missing ages (NaN): rows 6, 7
   - Missing heights/weights: scattered throughout

2. **Model's Solution**:
   - Successfully removed invalid ages
   - Attempted standardization (values near 0)
   - Some issues with imputation and index handling

3. **Grading Feedback**:
   - Partial success (0.65/1.0)
   - Strong on filtering and standardization
   - Needs improvement on imputation and data integrity

## �🔍 Future Enhancements

Potential areas for expansion:
- Additional data cleaning scenarios
- More complex edge cases
- Different data types
- Performance optimization metrics
- Code style evaluation