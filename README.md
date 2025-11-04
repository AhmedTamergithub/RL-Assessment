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

- Expected pass rate: 20-35%
- Passing score threshold: ≥ 0.9
- Common failure points:
  - Improper handling of NaN values
  - Incorrect standardization
  - Index management issues
  - Data integrity violations

## 🔧 Usage

```bash
# Ensure GEMINI_API_KEY is set
export GEMINI_API_KEY='your-key-here'

# Run the assessment
python assessment.py
```

## 🔍 Future Enhancements

Potential areas for expansion:
- Additional data cleaning scenarios
- More complex edge cases
- Different data types
- Performance optimization metrics
- Code style evaluation