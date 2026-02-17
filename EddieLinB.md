# Week 3 Assignment: Heart Disease Dataset Analysis

## I 320D: Data Science for Biomedical Informatics | Spring 2026

### 📋 Assignment Version B

---

## 🎯 This Week's Mantra

> **"Every Column Tells a Story"**

In this assignment, you'll apply the 10-Point Data Inspection to a real-world healthcare dataset focused on heart disease diagnosis. By the end, you'll understand not just *what* the data contains, but *why* each variable matters for clinical decision-making.

---

## Learning Objectives

By completing this assignment, you will be able to:

1. ✅ Apply the systematic 10-Point Inspection to a new healthcare dataset
2. ✅ Identify and classify feature types (continuous, discrete, categorical, ordinal)
3. ✅ Detect and document data quality issues (missing values, unexpected values)
4. ✅ Research and document clinical meaning for healthcare variables
5. ✅ Create meaningful data groupings based on clinical standards
6. ✅ Formulate answerable research questions about heart disease risk factors

---

## About the Dataset

**Dataset:** Heart Disease UCI (Combined)  
**Source:** UCI Machine Learning Repository / Kaggle  
**File:** `heart_disease_uci.csv`  
**Target Variable:** `num` (diagnosis of heart disease: 0 = no disease, 1-4 = presence of disease)

### Clinical Context

Heart disease remains the leading cause of death globally, accounting for approximately 17.9 million deaths annually according to the World Health Organization. This dataset combines patient records from four medical institutions: Cleveland Clinic, Hungarian Institute of Cardiology, University Hospital Zurich (Switzerland), and VA Long Beach Medical Center. Understanding these diagnostic variables is crucial for:

- Early identification of high-risk patients
- Understanding regional variations in heart disease presentation
- Clinical decision support systems
- Risk stratification and preventive care planning

---

## Getting Started

First, load the dataset and import your libraries:

```python
# Import libraries
import pandas as pd
import numpy as np

# Load the dataset
#Prints first 5 rows of our dataset
df = pd.read_csv('heart_disease_uci.csv')
# Display first few rows to confirm it loaded
#Prints first 5 rows of our dataset
df = pd.read_csv('heart_disease_uci.csv')

df.head(5)
```

---

## Part 1: The 10-Point Data Inspection (40 points)

Complete each inspection step and document your findings.

### Step 1: Shape (4 points)

**Your Code:**
```python
#Point 1 - Shape
df.shape

print(f'We have {df.shape[0]} rows and {df.shape[1]} columns in our dataset')
print(f'Our total number of data points is {df.shape[0] * df.shape[1]} points')
```

**Your Findings:**
Within the dataset there are 920 rows (patients) and then 16 different columns, all representing different statistics relating to the patients. This gives us 14720 data points across the dataset.

The columns are 'id' = id/index of the row, 'age' = age of the patient, 'dataset' = what hospital the data was sourced from, 'cp' = what kind of chest pain that the patient is experiencing, 'trestbps' = resting blood pressure in mm Hgs when admitted to the hospital, 'chol' = cholesterol levels in the patient, 'fbs' = fasting blood sugar over 120 mg/dl, 'restecg' = resting electrocardiographic results, 'thalach' = maximum heart rate achieved, 'exang' = exercise induced angina, 'oldpeak' = ST depression induced by exercise relative to rest, 'slope' = the slope of peak exercise ST segment, 'ca' = number of major vessels affected by fluroscopy, 'thal' = defect type, and 'num' = predicted attribute


### Step 2: Column Names (4 points)

**Your Code:**
```python
#Point 2 - Column Names
print("Column Names:")
print(df.columns.tolist())

print(f"\n Total columns: {len(df.columns)}")
```

**Your Findings:**
The columns are 'id', 'age', 'sex', 'dataset', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal', and 'num'.

The columns that need additional research to understand are dataset, cp, trestbps, fbs, restecg, thalch, exang, oldpeak, thal, and num.

---

### Step 3: Data Types (4 points)

**Your Code:**
```python
#Point 3 - Data Types
print("Data Types:")
print(df.dtypes)

print("\n" + "="*50)
print("Data Type Summary:")
print(df.dtypes.value_counts())
```

**Your Findings:**
The columns that are numeric are id, age, trestbps, chol, thalch, oldpeak, ca, and num. 

The categorical categories are sex, dataset, cp, restecg, slope, fbs,and thal.

None of the data types seem incorrect.

---

### Step 4: First Look (4 points)

**Your Code:**
```python
# Point 4 - Head
print("First 5 Rows:")
df.head()
```

**Your Findings:**
The actual values look pretty consistent and realistic. 

Most of the values are human readable such as sex, age, cp, slope, etc., but certain ones such as num and origin are confusing to those who aren't familiar with the dataset.

Some placeholder values within the dataset are 0s within columns where it would be impossible such as chol and trestbps, other columns use Null/NaN for placeholders/missing values such as ca, thal, slope, fbs, oldpeak, tresbps, thalch, exang, and chol.

---

### Step 5: Last Look (4 points)

**Your Code:**
```python
# Point 5 - Tail
print("Last 5 Rows:")
df.tail()
```

**Your Findings:**
The data ends decently cleanly, with all columns existing, but there are lots of missing values.

The last rows are pretty consistent with the first rows in terms of similar values and variety, but there are slightly more hypertension present patients.

There are a lot more missing values in the last rows compared to the first because the first had 0 while the worst of the last 5 rows is missing 7 columns.

---

### Step 6: Memory Usage (4 points)

**Your Code:**
```python
# Point 6 - Memory Usage
print("Memory Usage by Column:")
print(df.memory_usage(deep=True))

# Total memory in MB
total_memory_mb = df.memory_usage(deep=True).sum() / 1e6
print(f"\n Total Memory Usage: {total_memory_mb:.2f} MB")
```

**Your Findings:**
The dataset uses 0.42 MB of storage. 
This is a small dataset by data science standards because it is easily runnable across softwares and it performs well in excel and can be used with simple code that doesn't need optimization without any drawbacks.

---

### Step 7: Missing Values (4 points)

**Your Code:**
```python
# Point 7 - Missing Values
print("Missing Values by Column:")

# Calculate raw counts and percentages
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

# Combine them into a single summary table
missing_summary = pd.DataFrame({
    'Missing Values': missing,
    'Percentage (%)': missing_pct
})

# Display the table, rounding percentages to 2 decimal places for cleanliness
print(missing_summary.round(2))

print("\n" + "="*50)

# Calculate total missing values
total_missing = missing.sum()
print(f" Total Missing Values: {total_missing}")

# Check condition
if total_missing == 0:
    print("Great! No missing values - complete dataset!")
else:
    print(f" {total_missing} missing values need attention")
```

**Your Findings:**
The columns missing values are trestbps, chol, fbs, restecg, thalch, exang, oldpeak, oldpeak, slope, ca, thal.

The percentages are ca: 66.41%, thal: 52.83%, slope: 33.59%, fbs: 9.78%, oldpeak: 6.74%, trestbps: 6.41%, thalch: 5.98%, exang: 5.98%, chol: 3.26%,and restecg: 0.22%.

The columns that have the most missing values are ca, thal, and slope. Looking at the dataset these are missing mostly from the rows that originate from all the hospitals other than Cleveland. This could be due to these tests being more expensive and these other hospitals could lack the resources to commonly do these tests.

---

### Step 8: Duplicates (4 points)

**Your Code:**
```python
# Point 8 - Duplicates
duplicate_count = df.duplicated().sum()
duplicate_pct = (duplicate_count / len(df)) * 100

print(f" Duplicate Rows: {duplicate_count:,}")
print(f" Percentage: {duplicate_pct:.2f}%")

if duplicate_count > 0:
    print(f"\n Warning: {duplicate_pct:.2f}% of rows are duplicates!")
    print("   This needs investigation in Week 4 (Data Cleaning)")
``` 

**Your Findings:**
There are 0 duplicate rows in this dataset which means each patient ID is unique!

---

### Step 9: Basic Statistics (4 points)

**Your Code:**
```python
# Point 9 - Descriptive Statistics
print("Descriptive Statistics:")
df.describe().T
```

**Your Findings:**
- What is the age range in the dataset? 28 to 77
- What is the range of resting blood pressure (trestbps)? 0 to 200
- What is the range of serum cholesterol (chol)? 0 to 603
- What is the range of maximum heart rate achieved (thalch)? 60 to 202
  Resting blood pressure, rest bps, and chol of 0 all seem very unlikely and these 0s are most likely due to missing values and 0 being a placeholder.
---

### Step 10: Unique Counts (4 points)

**Your Code:**
```pyt# Point 10 - Unique Values
print("Unique Values per Column:")
unique_counts = df.nunique().sort_values()
print(unique_counts)hon

```

**Your Findings:**
The columns with very few unique values are sex, fbs, exang, with a count of 2, rectecg, slope, and thal, with counts of 3, cp, dataset, and ca, with counts of 4, and num with a count of 5.

The columns with many unique values are age with a count of 50, oldpeak with a count of 53, trestbps with a count of 61, thalch with a count of 119, chol with a count of 217, and oids with a count of 972.
- Which columns have many unique values (likely continuous or IDs)?

Yes the number of unique IDs matches the amount of rows.

---

## Part 2: Data Dictionary (20 points)

Complete the following data dictionary. For each column, you must:
1. **Research** the clinical meaning (these are standard cardiac assessment terms)
2. **Identify** the feature type (Continuous, Discrete, Categorical-Nominal, Categorical-Ordinal, Binary, Identifier)
3. **Document** the valid values/range you observe
4. **Note** any issues or questions

| Column | Description | Feature Type | Valid Values/Range | Notes/Issues |
|--------|-------------|--------------|-------------------|--------------|
| `id` | | | | |
| `age` | | | | |
| `sex` | | | | |
| `dataset` | | | | |
| `cp` | | | | |
| `trestbps` | | | | |
| `chol` | | | | |
| `fbs` | | | | |
| `restecg` | | | | |
| `thalch` | | | | |
| `exang` | | | | |
| `oldpeak` | | | | |
| `slope` | | | | |
| `ca` | | | | |
| `thal` | | | | |
| `num` | | | | |

### Clinical Research Questions for Version B

Answer these questions based on your research (you may need to use Google):

**1. What is hypertension? According to current American Heart Association guidelines, what are the blood pressure thresholds for normal, elevated, Stage 1 hypertension, and Stage 2 hypertension?**

Your answer: Hypertension, also known as high blood pressure is when the force of your blood pressing against your arteries is too high. Hypertension can lead to heart issues, strokes heart disease, and various other medical issues. Normal blood pressure is considered to be when systolic mm Hg is less than 120 and diastolic mm Hg is less than 80. Elevated blood pressure is when systolic mm Hg is between 120-129 and diastolic mm Hg is less than 80. Stage 1 hypertension is when systolic mm Hg is between 130-139 and diastolic mm Hg is between 80-89. Stage 2 hypertension is when systolic mm Hg is 140 or higher and diastolic mm Hg is higher than 90.

---

**2. What is resting ECG (restecg)? What do "normal," "ST-T abnormality," and "left ventricular hypertrophy" findings indicate about heart health?**

Your answer: Resting ECG (restecg) is a resting 12 lead cardiography test that detects abnormalities such as arrhythmias, coronary heart disease, and other conditions. An ECG takes a snapshot of the electrical activity of your heart. A normal finding means that the timing and strength of a heart falls within the normal range, while this is a good finding it doesn't rule out all heart conditions. ST-T abnormalities are when there is a change in the ST segment and the T wave, with the heart muscle resetting in between beats. The most common causes for this are Ischemia where the heart isn't getting enough oxygen/blood due to clogged arteries, electrolyte imbalances due to calcium and potassium, or medication side effects. Left Ventricular Hypertrophy/LVH is when the muscle wall of the main pumping chamber/left ventricle becomes thicker/enlarged. The causes of this are most commonly high blood pressure where the increased pressure makes the heart have to work harder to pump blood causing the muscle to become larger, valve issues where blood is harder to squeeze through and move, and athletes heart where increased physical activity can cause the heart to become stronger due to long term endurance training (marathon runners). These ECG findings are all indicators that can help us determine what conditions patients may be facing. 

---

**3. What is fasting blood sugar (fbs)? Why is the threshold of 120 mg/dl clinically significant? What does elevated fasting blood sugar indicate about diabetes risk?**

Your answer: Fasting blood sugar is a test where patient's blood sugar is measured after they haven't consumed anything other than water for 8-12 hours. A normal range is considered to be less than 100 mg/dl, Prediabetes 100-125mg/dl, and Diabetes is considered to be 126 mg/dl. 120 mg/dl places a patient very firmly in the range of prediabetes, with symptoms most likely arising already. Elevated fasting blood sugar while not diabetes yet already causes damage to the body with the lining of blood vessels, neuropathy, retinopathy, kidney damage, and loss of beta cells already possibly occuring within the body. 

---

**4. What does the number of major vessels colored by fluoroscopy (ca) tell us? Why might having more blocked vessels indicate worse heart disease?**

Your answer: Fluoroscopy shows doctors how well blood actually flows throughout the body, in a healthy heart they should get a score of 3 meaning that the dye is flowing and apparent in all 3 major blood vessels (Left Anterior Descending (LAD), Left Circumflex (LCX), and Right Coronary Artery (RCA). Scores lower than 3 means that one is blocked the patient is at risk of heart disease/failure. Most of the time heart disease starts in one area/one vessel meaning that if multiple are blocked, the disease is very widespread or the patient may be undergoing multiple conditions. The body cannot function healthily if even one of these vessels is damaged/at risk.

---

## Part 3: Data Validation (15 points)

### 3.1 Blood Pressure Validation (5 points)

Write code to check:
- How many blood pressure readings are 0? (This is clinically impossible for a living patient!)
- How many readings are below 80 mmHg? Above 200 mmHg?
- What might a value of 0 represent in this dataset?

**Your Code:**
```python
# Check for extreme outliers
impossible_high = df[df['trestbps'] > 250]
impossible_low = df[(df['trestbps'] < 60) & (df['trestbps'] != 0)]

print(f"Values > 250: {len(impossible_high)}")
print(f"Values < 60 (excluding 0): {len(impossible_low)}")

# Create a comparison table
cols_to_check = ['trestbps', 'chol', 'oldpeak', 'ca']

for col in cols_to_check:
    official_nulls = df[col].isnull().sum()
    zero_counts = (df[col] == 0).sum()
    print(f"Column: {col}")
    print(f"  - .isnull() reports: {official_nulls}")
    print(f"  - Zeros found:      {zero_counts}")
    print("-" * 25)

```

**Your Findings:**

- Are there any impossible blood pressure values?
There are 0 impossible blood pressure values ie none over 250 or under 60.
- How should values of 0 be treated - as missing data or as valid values?
Values of 0 can be viewed by a case by case basis for interpretation, but in the case of "trestbps" they mean missing data due to them being an impossible value.
- Does this match what `.isnull()` reports for this column?
For trestbps it does not match it because there are 59 cases of .isnull results and only one zero. This could be attributed to human error and a result of the dataset being compiled from 4 different hospitals.
---

### 3.2 Cholesterol Validation (5 points)

Write code to examine cholesterol values closely. Check for any unusual values including zeros.

**Your Code:**
```python
import pandas as pd
import numpy as np

# 1. Basic counts of missingness
official_nulls = df['chol'].isnull().sum()
zero_values = (df['chol'] == 0).sum()

# 2. Statistical summary of non-zero data
# We filter out zeros to get a true sense of the "real" distribution
real_chol = df[df['chol'] > 0]['chol']

print(f"--- Cholesterol Health Check ---")
print(f"Official NaN values: {official_nulls}")
print(f"Zero values (Hidden Nulls): {zero_values}")
print(f"Total 'Missing' Data: {official_nulls + zero_values}")
print(f"\n--- Distribution of Valid Data ---")
print(f"Min (Non-zero): {real_chol.min()}")
print(f"Max:            {real_chol.max()}")
print(f"Median:         {real_chol.median()}")

# 3. Identify extreme outliers (Physiologically suspicious)
# Values over 500 are rare and represent severe hypercholesterolemia
extreme_high = df[df['chol'] > 500]
print(f"\nRecords with Chol > 500: {len(extreme_high)}")

# 4. Final check: Does .isnull() match reality?
matches = (official_nulls == (official_nulls + zero_values))
print(f"\nDoes .isnull() report all missing data? {matches}")
```

**Your Findings:**
- Did you find any cholesterol values of 0?
There were 172 cholesterol values of 0 and 30 null values, giving us 202 instances of missing cholesterol.
- Is a cholesterol level of 0 clinically possible?
A cholesterol level of 0 is not clinically possible unless through very very rare circumstances with genetic disorders occurring.
- How many such impossible values exist?
The impossible values that exist within our dataset for cholesterol are the 172 values of 0 and then 4 records that have chol levels over 500.
- What would happen if you calculated the mean cholesterol without handling this?
If I calculated the mean cholesterol without handling this it could skew it lower or higher than it already is due to the outliers or 0 values. With the high prevalence of 0s it would most likely skew it lower than it actually is.
---

### 3.3 Multi-Source Data Validation (5 points)

This dataset combines data from four different medical centers. Write code to:
1. Count patients from each source (dataset column)
2. Check if missing data patterns differ by source

**Your Code:**
```python
import pandas as pd

# Load the dataset
df = pd.read_csv('heart_disease_uci.csv')

# 1. Count patients from each source
print("--- Patient Counts by Source ---")
source_counts = df['dataset'].value_counts()
print(source_counts)

# Identify the smallest contributor
fewest_source = source_counts.idxmin()
fewest_count = source_counts.min()
print(f"\nSmallest contributor: {fewest_source} ({fewest_count} patients)")

# 2. Check missing data patterns by source (Calculated as Percentages)
print("\n--- Missing Data Percentages by Source (%) ---")
missing_pct_by_source = df.groupby('dataset').apply(lambda x: x.isnull().mean() * 100).round(2)
print(missing_pct_by_source)
```

**Your Findings:**

- Which medical center contributed the most patients?
The medical center that contributed the most patients was Cleveland.
- Which medical center contributed the fewest?
The medical center that contributed the least was Switzerland.
- Do certain columns have more missing data in certain sources? Which ones?
The more complex columns have more missing data in sources like VA Long Beach and Switzerland. The columns with the most missing values in order are ca, thal, slope, thalc, restecg, fbs, exang, oldpeak, and chol.
- Why might data completeness vary across medical institutions?
Data completeness varies across institutions due to a variety of reasons such as the resources available within the hospitals. Cleveland may have been the most equipped hospital, allowing them to conduct tests easily, while the others lacked in certain areas possibly not having the resources or training to do these tests. Another factor is the staff at each hospital, levels of training and proficiency vary and some of the hospitals may not have enough staff to conduct these tests.
---

## Part 4: Create Clinical Blood Pressure Groups (10 points)

Create a new column called `bp_category` that categorizes patients based on their resting blood pressure according to American Heart Association guidelines.

### Version B: Blood Pressure Categories

Use these categories based on AHA hypertension guidelines:

| BP Category | Systolic Range (trestbps) | Clinical Significance |
|-------------|---------------------------|----------------------|
| Missing/Invalid | 0 or NaN | Data quality issue - cannot classify |
| Normal | < 120 | Healthy blood pressure |
| Elevated | 120-129 | At risk for hypertension |
| Stage 1 HTN | 130-139 | Hypertension Stage 1 |
| Stage 2 HTN | ≥ 140 | Hypertension Stage 2 - requires treatment |

**Note:** HTN = Hypertension. Real BP classification uses both systolic AND diastolic, but we only have systolic (trestbps) in this dataset.

### Your Code:

```python
# Create the bp_category column
# IMPORTANT: You must handle values of 0 appropriately - they should be "Missing/Invalid"
# You can use a custom function with .apply() OR pd.cut()
import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv('heart_disease_uci.csv')

# 2. Define the exact conditions based on the AHA guidelines
conditions = [
    # Missing or 0
    df['trestbps'].isna() | (df['trestbps'] == 0),
    # Normal: < 120
    df['trestbps'] < 120,
    # Elevated: 120-129
    (df['trestbps'] >= 120) & (df['trestbps'] < 130),
    # Stage 1 HTN: 130-139
    (df['trestbps'] >= 130) & (df['trestbps'] < 140),
    # Stage 2 HTN: >= 140
    df['trestbps'] >= 140
]

# 3. Define the corresponding categories
choices = [
    'Missing/Invalid',
    'Normal',
    'Elevated',
    'Stage 1 HTN',
    'Stage 2 HTN'
]

# 4. Apply the conditions to create the new column
df['bp_category'] = np.select(conditions, choices, default='Unknown')

# 5. Check the distribution
print(df['bp_category'].value_counts())

```

### Verify your groupings worked:

```python
# Show counts per BP category
print(df['bp_category'].value_counts())
```

### Calculate heart disease rate by blood pressure category:

```python
# Create a binary indicator for heart disease (num > 0 means disease present)
# Then calculate the percentage of patients with heart disease in each BP category
# Exclude the "Missing/Invalid" category from your interpretation
import pandas as pd
import numpy as np

# --- STEP 1: CREATE THE CATEGORIES ---

# 1. Load the dataset
df = pd.read_csv('heart_disease_uci.csv')

# 2. Define the exact conditions based on the AHA guidelines
conditions = [
    # Missing or 0
    df['trestbps'].isna() | (df['trestbps'] == 0),
    # Normal: < 120
    df['trestbps'] < 120,
    # Elevated: 120-129
    (df['trestbps'] >= 120) & (df['trestbps'] < 130),
    # Stage 1 HTN: 130-139
    (df['trestbps'] >= 130) & (df['trestbps'] < 140),
    # Stage 2 HTN: >= 140
    df['trestbps'] >= 140
]

# 3. Define the corresponding categories
choices = [
    'Missing/Invalid',
    'Normal',
    'Elevated',
    'Stage 1 HTN',
    'Stage 2 HTN'
]

# 4. Apply the conditions to create the new column
df['bp_category'] = np.select(conditions, choices, default='Unknown')

# 5. Check the distribution
print("--- Blood Pressure Categories Distribution ---")
print(df['bp_category'].value_counts())
print("\n")


# --- STEP 2: CALCULATE DISEASE RATES ---
# (Notice we DO NOT reload the CSV here, we just keep using 'df')

# 1. Create a binary indicator for heart disease (1 = disease, 0 = no disease)
df['has_disease'] = (df['num'] > 0).astype(int)

# 2. Exclude the "Missing/Invalid" category
valid_bp_df = df[df['bp_category'] != 'Missing/Invalid']

# 3. Calculate the percentage of patients with heart disease in each BP category
disease_rates = valid_bp_df.groupby('bp_category')['has_disease'].mean() * 100

# 4. Reorder the categories logically for interpretation
category_order = ['Normal', 'Elevated', 'Stage 1 HTN', 'Stage 2 HTN']
disease_rates = disease_rates.reindex(category_order)

print("--- Percentage of Patients with Heart Disease by BP Category ---")
print(disease_rates.round(2).astype(str) + '%')
```

### Analysis Questions:

**1. How many patients are in each blood pressure category? How many have missing/invalid blood pressure readings?**

Your answer: There are 161 patients in the normal category, 211 in the elevated category, 177 in the Stage 1 HTN, 311 in the Stage 2 HTN, and 60 missing values. These missing values have missing/invalid blood pressure readings.

---

**2. What is the heart disease rate (percentage) for each blood pressure category (excluding invalid)?**

Your answer: The heart disease percentage rate is 54.04% for people with normal blood pressure levels, 49.29% for those with elevated blood pressure levels, 48.59% with those with Stage 1 HTN,and 61.74% with Stage 2 HTN. 

---

**3. Does heart disease prevalence increase with blood pressure as expected? What does this suggest about the relationship between hypertension and heart disease?**

Your answer: Heart disease prevalence doesn't increase with blood pressure until it hits Stage 2 HTN according to our data set. It did not reflect what I expected due to me expecting a correlation between hypertension and heart disease.

---

## Part 5: Research Questions (15 points)

### 5.1 Write Three Answerable Questions (9 points)

Write three questions that THIS dataset can answer. Remember: the data can show relationships and patterns, but cannot prove causation.

**Your questions must explore these specific areas:**

1. **A question about resting ECG results (restecg) and heart disease:**
How do people with normal resting ECGs compare to those with abnormal ECGs caused by ST-T wave abnormalities or left ventricle hypertrophy in heart disease diagnoses and severity?

---

2. **A question comparing the four medical centers (dataset):**
Are patients from certain medical centers more likely to have heart disease or hypertension than others?

---

3. **A question about the combination of diabetes indicator (fbs) AND exercise-induced angina (exang):**
Are patients with high fbs and exercise induced angina more likely to experience heart disease than patients with one or neither?

---

### 5.2 Identify One Question the Data CANNOT Answer (3 points)

Write one question about **patient lifestyle or family history** that this dataset cannot answer, and explain why.

**Question:**
Are patients of a certain race/culture more likely to develop heart disease?

**Why it cannot be answered with this data:**
This data doesn't provide race or culture data, only providing rough locations due to the names of the hospitals. This question would be connected to more specific demographic information possibly even delving into economic status as well as race/culture data.

---

### 5.3 Grouping Analysis (3 points)

Answer this question using a groupby analysis:

**"What is the average cholesterol level (chol) for patients with vs. without heart disease?"**

(Note: You'll need to create a binary heart disease indicator and handle cholesterol values of 0!)

**Your Code:**
```python
import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv('heart_disease_uci.csv')

# 2. Handle invalid cholesterol values of 0 (replace with NaN)
df['chol'] = df['chol'].replace(0, np.nan)

# 3. Create a binary heart disease indicator (1 = Disease, 0 = No Disease)
df['has_disease'] = (df['num'] > 0).astype(int)

# 4. Perform the groupby analysis
avg_chol = df.groupby('has_disease')['chol'].mean()

print("Average Cholesterol by Heart Disease Status:")
print(avg_chol.round(2))
```

**Your Interpretation:**

Do patients with heart disease have higher average cholesterol than those without? Is the difference what you expected? What might confound this simple comparison?
People with heart disease have higher average cholesterol with them having 254.01 mg/dl on average compared to the 240.16 mg/dl of those without heart disease. I wasn't sure of what level the difference would be, but I believed that there would be a difference. Somethings that could confound this comparison are age of patients, diet, missing values in datasets, or biological sex among many other variables.

---

## Part 6: Target Variable Analysis (Bonus - 5 points)

The `num` column is our **target variable** - what we're trying to predict. It has 5 possible values (0-4) representing the severity of heart disease narrowing. Analyze its distribution.

**Your Code:**
```python
# Show the count and percentage for each value of num

# Also create a binary version (0 = no disease, 1 = disease present) and show its distribution
import pandas as pd

# Load the dataset
df = pd.read_csv('heart_disease_uci.csv')

# 1. Distribution of the original 'num' column (Severity 0-4)
print("--- Distribution of 'num' (0-4 Severity) ---")
num_counts = df['num'].value_counts().sort_index()
num_percentages = (df['num'].value_counts(normalize=True) * 100).sort_index()

num_dist = pd.DataFrame({
    'Count': num_counts,
    'Percentage (%)': num_percentages.round(2)
})
print(num_dist)


print("\n" + "="*50 + "\n")


# 2. Create a binary version (0 = no disease, 1 = disease present)
df['disease_binary'] = (df['num'] > 0).astype(int)

# Show its distribution
print("--- Distribution of Binary Disease Indicator ---")
binary_counts = df['disease_binary'].value_counts().sort_index()
binary_percentages = (df['disease_binary'].value_counts(normalize=True) * 100).sort_index()

binary_dist = pd.DataFrame({
    'Count': binary_counts,
    'Percentage (%)': binary_percentages.round(2)
})
binary_dist.index = ['0 (No Disease)', '1 (Disease Present)']
print(binary_dist)
```

There are 411 patients with no disase, being 44.67% of the dataset, while 509 patients have disease present being 55.33% of the dataset.

### Bonus Questions:

**1. What percentage of patients in this dataset have some form of heart disease (num > 0)?**

Your answer: 55.33% of patients have some form of heart disease.

---

**2. Is this dataset balanced or imbalanced between disease/no disease?**

Your answer: This dataset is balanced between disease and no disease, with a less than 10% difference in distribution.

---

**3. Looking at the 5 severity levels (0-4), which level is most common? Is this distribution what you'd expect clinically?**

Your answer: Level 0 (normal) is the most common severity level. We would expect this clinically because typically there are more healthy patients, and as severity increases there are less of each category.

---

**4. How might the multi-severity nature of the target (0-4) versus a simple binary (disease/no disease) affect how you'd approach a machine learning problem?**

Your answer: The multi-severity nature of the target compared to a binary system would affect machine learning problems by increasing the complexity of the algorithm needed to train the machine learning. With a binary system it is much easier to train a model with them only having 2 possible results, while a 5 outcome model will have much differing levels of error. If a patient is predicted wrongly on the binary system it has only one outcome of false positives while the 5 outcome model has much more possible false results, affecting patients more if for example a severe patient was predicted to be a 0 or 1, causing less tests to be ran.

---

## Submission Checklist

Before submitting, verify you have completed:

- [ ] **Part 1:** All 10 inspection steps with code AND written findings
- [ ] **Part 2:** Complete data dictionary with all 16 columns filled in
- [ ] **Part 2:** Answered all 4 clinical research questions
- [ ] **Part 3:** All 3 validation checks with code and answers
- [ ] **Part 4:** Created `bp_category` column using **AHA Blood Pressure Categories**
- [ ] **Part 4:** Calculated heart disease rate by BP category with interpretation
- [ ] **Part 5:** Three research questions (restecg, medical centers, fbs+exang)
- [ ] **Part 5:** One unanswerable question about lifestyle/family history
- [ ] **Part 5:** Cholesterol by heart disease status groupby analysis
- [ ] **Bonus (Optional):** Target variable analysis

---

## Grading Rubric

| Component | Points | Requirements for Full Credit |
|-----------|--------|------------------------------|
| Part 1: 10-Point Inspection | 40 | All 10 steps complete with working code AND thoughtful written analysis |
| Part 2: Data Dictionary | 20 | All 16 columns documented with correct feature types and clinical research |
| Part 3: Data Validation | 15 | All validation checks complete with working code and insightful answers |
| Part 4: BP Categories | 10 | Working code that creates correct groups (including invalid handling) AND meaningful interpretation |
| Part 5: Research Questions | 15 | Three good questions in specified areas, one clear limitation, groupby analysis complete |
| **Bonus:** Target Analysis | +5 | Thoughtful analysis of target variable distribution with clinical connection |
| **Total** | 100 (+5 bonus) | |

---

## Hints (Read Before You Get Stuck!)

### ⚠️ Common Pitfalls:

1. **Values of 0 that aren't really zero**
   - Some columns (like `trestbps` and `chol`) have 0 values that represent missing data, not actual measurements
   - The `.isnull()` method won't detect these!
   - Always examine your data with `value_counts()` and check if extreme values make clinical sense

2. **Missing data varies by source**
   - The four medical centers have different data completeness
   - Some columns like `ca`, `thal`, and `slope` have extensive missing data
   - Understanding WHY data is missing is as important as knowing that it's missing

3. **The target variable has multiple levels**
   - `num` ranges from 0-4, not just 0-1
   - For many analyses, you'll want to convert this to binary (disease present/absent)

4. **Sex is coded as text**
   - Unlike some versions of this dataset, sex is "Male"/"Female" not 0/1

### 💡 Pro Tips:

- Use `value_counts(dropna=False)` to see if there are any null values
- When something seems weird (like 0 blood pressure), investigate it—don't assume it's valid
- Cross-reference columns: do the patterns make clinical sense together?
- Read the original UCI documentation for additional context
- Always state what the data CAN tell you vs. what would require additional information

---

## Useful Resources

- **UCI Repository Page:** https://archive.ics.uci.edu/ml/datasets/heart+disease
- **AHA Blood Pressure Categories:** https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings
- **ECG Interpretation Basics:** https://litfl.com/ecg-library/
- **Cholesterol Levels Guide:** https://www.heart.org/en/health-topics/cholesterol/about-cholesterol
- **Pandas Documentation:** https://pandas.pydata.org/docs/

---

*Remember: "Every Column Tells a Story" - your job is to figure out what that story is!*

---

**Due Date:** [See Canvas]

**Submission:** Upload your completed Jupyter notebook (.ipynb) to Canvas
