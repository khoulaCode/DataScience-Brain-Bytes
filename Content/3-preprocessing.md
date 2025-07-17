# Data Preprocessing 
Data preprocessing is a crucial step in the data science pipeline. It involves transforming raw data into a clean, structured, and usable format for building machine learning models. Without proper preprocessing, models may perform poorly or yield inaccurate results.

This guide will walk you through essential preprocessing techniques, providing you with the knowledge to handle common data quality issues and prepare your datasets effectively.

## 1. Handling Null (Missing) Values
Missing values are a common problem in real-world datasets. They can arise due to various reasons, such as data entry errors, data corruption, or simply information not being collected. Dealing with them appropriately is vital.

### 1.1. Identifying Null Values
Before you can handle missing values, you need to identify them.

Using Pandas isnull() or isna():
```py
import pandas as pd
import numpy as np

# Example DataFrame
data = {'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, 4, np.nan],
        'C': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

print("DataFrame with NaNs:")
print(df)

# Check for null values across the entire DataFrame
print("\nNull values across DataFrame:")
print(df.isnull())

# Count null values per column
print("\nCount of null values per column:")
print(df.isnull().sum())

# Percentage of null values per column
print("\nPercentage of null values per column:")
print(df.isnull().sum() / len(df) * 100)
```
### 1.2. Options to Deal with Null Values
#### a) Deletion
Deletion is the simplest method but can lead to significant data loss if not used carefully.

- Dropping Rows with Any Null Value (dropna()):
This removes any row that contains at least one missing value.
```py
df_dropped_rows = df.dropna()
print("\nDataFrame after dropping rows with any NaN:")
print(df_dropped_rows)
```
- Dropping Rows with All Null Values (dropna(how='all')):
This removes rows only if all values in that row are missing.
```py
data_all_nan = {'A': [1, np.nan], 'B': [np.nan, np.nan]}
df_all_nan = pd.DataFrame(data_all_nan)
df_all_nan.loc[2] = [np.nan, np.nan] # Add a row with all NaNs

print("\nOriginal DataFrame with a row of all NaNs:")
print(df_all_nan)

df_dropped_all_nan_rows = df_all_nan.dropna(how='all')
print("\nDataFrame after dropping rows with all NaNs:")
print(df_dropped_all_nan_rows)
```
- Dropping Columns with Any Null Value (dropna(axis=1)):
This removes any column that contains at least one missing value.
```py
df_dropped_cols = df.dropna(axis=1)
print("\nDataFrame after dropping columns with any NaN:")
print(df_dropped_cols)
```
- Dropping Columns with All Null Values (dropna(axis=1, how='all')):
This removes columns only if all values in that column are missing.
```py
data_all_nan_col = {'A': [1, 2, 3], 'B': [np.nan, np.nan, np.nan]}
df_all_nan_col = pd.DataFrame(data_all_nan_col)

print("\nOriginal DataFrame with a column of all NaNs:")
print(df_all_nan_col)

df_dropped_all_nan_cols = df_all_nan_col.dropna(axis=1, how='all')
print("\nDataFrame after dropping columns with all NaNs:")
print(df_dropped_all_nan_cols)
```
- Dropping Rows/Columns based on a Threshold (dropna(thresh=N)):
This keeps rows/columns that have at least N non-null values.
```py
# Keep rows with at least 4 non-null values
df_thresh = df.dropna(thresh=2) # In our example, all rows have at least 2 non-nulls
print("\nDataFrame after dropping rows with less than 2 non-nulls:")
print(df_thresh)
```
#### b) Imputation
Imputation involves filling missing values with estimated values. This is generally preferred over deletion as it preserves more data.

- **Filling with a Constant Value** (`fillna(value)`):
Useful when you have a specific value to replace NaNs with (e.g., 0 for counts, 'Unknown' for categories).
```py
df_filled_const = df.fillna(0)
print("\nDataFrame after filling NaNs with 0:")
print(df_filled_const)
```
- **Filling with Mean** (`fillna(df.mean())`):
Best for numerical data that is approximately normally distributed and without significant outliers.
```py
df_filled_mean = df.fillna(df.mean(numeric_only=True))
print("\nDataFrame after filling NaNs with column mean:")
print(df_filled_mean)
```
- **Filling with Median** (`fillna(df.median())`):
More robust to outliers than the mean, suitable for **skewed** numerical data.
```py
df_filled_median = df.fillna(df.median(numeric_only=True))
print("\nDataFrame after filling NaNs with column median:")
print(df_filled_median)
```
- **Filling with Mode** (`fillna(df.mode().iloc[0])`):
Most suitable for categorical data, but can also be used for numerical data. `iloc[0]` is used because `mode()` can return **multiple modes**.
```py
df_filled_mode = df.fillna(df.mode().iloc[0])
print("\nDataFrame after filling NaNs with column mode:")
print(df_filled_mode)

# Example for categorical data
cat_data = {'Color': ['Red', 'Blue', np.nan, 'Red', 'Green']}
df_cat = pd.DataFrame(cat_data)
df_cat_filled_mode = df_cat.fillna(df_cat['Color'].mode()[0])
print("\nCategorical DataFrame after filling NaNs with mode:")
print(df_cat_filled_mode)
```
- **Forward Fill** (`ffill()` or `fillna(method='ffill')`):
Propagates the last valid observation forward to next valid observation. Useful for time series data.
```py
df_ffill = df.fillna(method='ffill')
print("\nDataFrame after forward fill (ffill):")
print(df_ffill)
```
- **Backward Fill** (`bfill()` or `fillna(method='bfill')`):
Uses the next valid observation to fill the gap. Also useful for time series data.
```py
df_bfill = df.fillna(method='bfill')
print("\nDataFrame after backward fill (bfill):")
print(df_bfill)
```

## 2. Checking and Removing Duplicate Values
Duplicate values can skew your analysis and lead to **biased models**. It's important to identify and remove them.

### 2.1. Identifying Duplicate Values
- Using `duplicated()`:
This method returns a boolean Series indicating whether each row is a duplicate of a previous row.
```py
data_dup = {'A': [1, 2, 3, 2, 4],
            'B': ['X', 'Y', 'Z', 'Y', 'W'],
            'C': [10, 20, 30, 20, 40]}
df_dup = pd.DataFrame(data_dup)

print("\nOriginal DataFrame with duplicates:")
print(df_dup)

# Identify duplicate rows (keeping the first occurrence)
print("\nBoolean Series indicating duplicate rows:")
print(df_dup.duplicated())

# Show actual duplicate rows
print("\nActual duplicate rows (excluding first occurrence):")
print(df_dup[df_dup.duplicated()])

# Identify duplicate rows (keeping the last occurrence)
print("\nDuplicate rows (keeping last occurrence):")
print(df_dup[df_dup.duplicated(keep='last')])

# Identify all duplicate occurrences
print("\nAll duplicate occurrences:")
print(df_dup[df_dup.duplicated(keep=False)])

# Check for duplicates based on specific columns
print("\nDuplicates based on columns 'A' and 'B':")
print(df_dup[df_dup.duplicated(subset=['A', 'B'])])
```
### 2.2. Removing Duplicate Values
- Using `drop_duplicates()`:
This method removes duplicate rows from the DataFrame. By default, it keeps the first occurrence.
```py
# Remove duplicate rows (keeping the first occurrence)
df_no_dup = df_dup.drop_duplicates()
print("\nDataFrame after dropping duplicates (keeping first):")
print(df_no_dup)

# Remove duplicate rows (keeping the last occurrence)
df_no_dup_last = df_dup.drop_duplicates(keep='last')
print("\nDataFrame after dropping duplicates (keeping last):")
print(df_no_dup_last)

# Remove all duplicate occurrences
df_no_dup_all = df_dup.drop_duplicates(keep=False)
print("\nDataFrame after dropping all duplicate occurrences:")
print(df_no_dup_all)

# Remove duplicates based on specific columns
df_no_dup_subset = df_dup.drop_duplicates(subset=['A', 'B'])
print("\nDataFrame after dropping duplicates based on 'A' and 'B':")
print(df_no_dup_subset)
```
## 3. Removing Collinearity (Multicollinearity) from the Dataset
Multicollinearity occurs when independent variables in a regression model are highly correlated with each other. This can make it difficult to interpret the individual impact of each predictor and can lead to unstable model coefficients.

### 3.1. Identifying Collinearity
- **Correlation Matrix**:
A simple way to detect multicollinearity is by examining the correlation matrix between independent numerical variables. High absolute correlation coefficients (e.g., greater than 0.7 or 0.8) indicate potential multicollinearity.
```py
# Example DataFrame with collinearity
data_collinear = {'Feature1': [10, 12, 15, 11, 13],
                  'Feature2': [20, 24, 30, 22, 26], # Highly correlated with Feature1 (2 * Feature1)
                  'Feature3': [5, 6, 7, 5, 6],
                  'Target': [100, 120, 150, 110, 130]}
df_collinear = pd.DataFrame(data_collinear)

print("\nOriginal DataFrame for collinearity example:")
print(df_collinear)

# Calculate the correlation matrix
correlation_matrix = df_collinear[['Feature1', 'Feature2', 'Feature3']].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)
```
**Observation**: 'Feature1' and 'Feature2' have a correlation of 1.0, indicating perfect multicollinearity.


### 3.2. Dealing with Collinearity
#### a) Feature Selection (Dropping Correlated Features)
The most common approach is to remove one of the highly correlated variables. Choose the one that is less important for your analysis or has less predictive power.
```py
# Drop 'Feature2' as it's perfectly correlated with 'Feature1'
df_no_collinear = df_collinear.drop('Feature2', axis=1)
print("\nDataFrame after dropping 'Feature2' to reduce collinearity:")
print(df_no_collinear)
```
#### b) Principal Component Analysis (PCA)
PCA is a dimensionality reduction technique that transforms a set of correlated variables into a smaller set of uncorrelated variables called principal components. These components capture most of the variance in the original data.

## 4. Data Normalization and Scaling
Normalization (Min-Max Scaling) and Standardization (Z-score normalization) are techniques used to transform numerical features so they have a *similar scale*. This is crucial for many machine learning algorithms (e.g., K-Means, SVM, Neural Networks) that are **sensitive to the magnitude of input features**.

### 4.1. When to Use
- Algorithms sensitive to feature scales: K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Logistic Regression, Neural Networks, Gradient Descent-based algorithms.

- Distance-based algorithms: When the distance between data points is calculated (e.g., Euclidean distance), features with larger ranges will dominate the calculation.

- Regularization techniques: L1/L2 regularization penalize large coefficients, which can be disproportionately affected by unscaled features.

### 4.2. Methods
#### **a) Min-Max Scaling (Normalization)**
Scales features to a fixed range, usually between 0 and 1.

- Formula: 

$$
X_{scaled} = \frac {X− X_{min}} {X_{max} − X_{min}}
$$

Sensitive to outliers.
```py
from sklearn.preprocessing import MinMaxScaler

data_scale = {'FeatureA': [10, 20, 30, 40, 50],
              'FeatureB': [1, 2, 3, 4, 5]}
df_scale = pd.DataFrame(data_scale)

print("\nOriginal DataFrame for scaling example:")
print(df_scale)

scaler_minmax = MinMaxScaler()
df_minmax_scaled = pd.DataFrame(scaler_minmax.fit_transform(df_scale), columns=df_scale.columns)
print("\nDataFrame after Min-Max Scaling:")
print(df_minmax_scaled)
```
#### **b) Standardization (Z-score Normalization)**
Scales features to have a mean of 0 and a standard deviation of 1.

- Formula: 

$$
X_{scaled}=
\frac{X−\mu}{\sigma}
$$

  Where $\mu$ is the mean and $\sigma$ is the standard deviation.

Less sensitive to outliers than Min-Max scaling.
```py
from sklearn.preprocessing import StandardScaler

scaler_standard = StandardScaler()
df_standard_scaled = pd.DataFrame(scaler_standard.fit_transform(df_scale), columns=df_scale.columns)
print("\nDataFrame after Standardization:")
print(df_standard_scaled)
```
#### **c) Robust Scaling**
Scales features using the interquartile range (IQR) and median, making it robust to outliers.

**Formula:**

$$
X_{scaled}=\frac{X−Median}{IQR}
$$
```py
from sklearn.preprocessing import RobustScaler

scaler_robust = RobustScaler()
df_robust_scaled = pd.DataFrame(scaler_robust.fit_transform(df_scale), columns=df_scale.columns)
print("\nDataFrame after Robust Scaling:")
print(df_robust_scaled)
```
#### **d) Log Transformation**
The log transformation (natural logarithm, base 10, or base 2) is effective for positively skewed data. It compresses the larger values and expands the smaller values, making the distribution more symmetrical.

- **Formula:** $log(x)$ or $log(1+x)$ (to handle zero values, as $log(0)$ is undefined).

- **When to use:** When data is heavily positively skewed and contains only positive values. $log(1+x)$ is preferred if zeros are present.

- **Effect:** Reduces the impact of extreme values.
```py

# Apply Log Transformation
df['Log_Transformed_Feature'] = np.log1p(df['Skewed_Feature']) # log1p handles 0 values (log(1+x))
```
#### **e) Square Root Transformation**
Similar to log transformation, the square root transformation is used for positively skewed data. It's less aggressive than log transformation.

- **Formula**: 

$sqrt(x)$

- **When to use:** When data is positively skewed and contains only non-negative values.

- **Effect**: Moderately reduces skewness.
```py
# Apply Square Root Transformation
df['Sqrt_Transformed_Feature'] = np.sqrt(df['Skewed_Feature'])
```
#### f) Box-Cox Transformation
The Box-Cox transformation is a family of power transformations that can transform non-normal dependent variables into a normal shape. It is more flexible than log or square root transformations as it considers a range of $lambda$ values.

Formula:

If $lambda \neq 0:$

$$
X_{scaled} =\frac{X^\lambda −1}{\lambda} 
$$

If $lambda=0:$

$$
X_{scaled} = log(X)
$$

- **When to use**: When data is positively skewed and contains only positive values. It automatically finds the best 
lambda value that normalizes the data.

- **Effect**: Can achieve a more normal distribution compared to fixed transformations.
```py
from scipy.stats import boxcox

# Assuming 'df' is your DataFrame with a skewed column 'Skewed_Feature'

# Apply Box-Cox Transformation
df['BoxCox_Transformed_Feature'], lambda_val = boxcox(df['Skewed_Feature']) # Returns transformed data and optimal lambda
```
## 5. Data Categorization (Encoding Categorical Data)
***Machine learning models typically require numerical input***. Categorical variables (e.g., 'Color', 'City') need to be converted into numerical representations.

### 5.1. Nominal vs. Ordinal Categorical Data
- **Nominal**: Categories have no intrinsic order (e.g., 'Red', 'Blue', 'Green').

- **Ordinal**: Categories have a meaningful order (e.g., 'Low', 'Medium', 'High'; 'Small', 'Medium', 'Large').

### 5.2. Methods
#### a) One-Hot Encoding
Creates new binary (0 or 1) columns for each category in a feature. Best for nominal categorical data.

Avoids implying an artificial order.

*Can lead to a high-dimensional dataset if there are many unique categories.*
```py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Sample DataFrame
df = pd.DataFrame({
    'color': ['red', 'green', 'blue', 'green', 'red', 'blue']
})

# OneHotEncoder requires 2D input and returns a NumPy array
ohe = OneHotEncoder(sparse=False)  # sparse=False returns a dense array
color_ohe = ohe.fit_transform(df[['color']])

# Create column names for one-hot encoded columns
ohe_columns = ohe.get_feature_names_out(['color'])

# Combine with original DataFrame
df_ohe = pd.DataFrame(color_ohe, columns=ohe_columns)
df_combined = pd.concat([df, df_ohe], axis=1)

print(df_combined)
```
#### b) Label Encoding
Assigns a unique integer to each category. Best for ordinal categorical data where the numerical order reflects the actual order.

Implies an order, so be careful with nominal data.
```py
from sklearn.preprocessing import LabelEncoder

# For 'Size' column (ordinal data)
le = LabelEncoder()
df['color_label'] = le.fit_transform(df['color'])
print("\nDataFrame after Label Encoding 'Size':")
print(df)

# To see the mapping
print("\nLabel Encoder mapping for 'Size':")
for i, item in enumerate(le.classes_):
    print(f"{item}: {i}")
```
#### c) Ordinal Encoding
Similar to Label Encoding but allows you to specify the order of categories.
```py
from sklearn.preprocessing import OrdinalEncoder

# Define the explicit order for 'Size'
size_order = ['Small', 'Medium', 'Large']
encoder_oe = OrdinalEncoder(categories=[size_order])

df_cat_encode['Size_Ordinal_Encoded'] = encoder_oe.fit_transform(df_cat_encode[['Size']])
print("\nDataFrame after Ordinal Encoding 'Size' with custom order:")
print(df_cat_encode)
```

## 6. Applying Functions to Create New Columns in DataFrame
You often need to derive new features from existing ones to capture more complex relationships or simplify existing data.

### 6.1. Using Existing Columns with Arithmetic Operations
```py
data_ops = {'Price': [100, 150, 200, 250],
            'Quantity': [2, 3, 1, 4]}
df_ops = pd.DataFrame(data_ops)

print("\nOriginal DataFrame for new column creation:")
print(df_ops)

# Create 'Total_Cost' column
df_ops['Total_Cost'] = df_ops['Price'] * df_ops['Quantity']
print("\nDataFrame with 'Total_Cost' column:")
print(df_ops)

# Create 'Price_Per_Item' column
df_ops['Price_Per_Item'] = df_ops['Total_Cost'] / df_ops['Quantity']
print("\nDataFrame with 'Price_Per_Item' column:")
print(df_ops)
```
### 6.2. Using Lambda Functions
Lambda functions are small, anonymous functions that can be used for quick operations.
```py
# Create a 'Discounted_Price' column using a lambda function
# Apply a 10% discount if Total_Cost > 300, else no discount
df_ops['Discounted_Price'] = df_ops.apply(lambda row: row['Total_Cost'] * 0.9 if row['Total_Cost'] > 300 else row['Total_Cost'], axis=1)
print("\nDataFrame with 'Discounted_Price' column (using lambda):")
print(df_ops)
```
### 6.3. Using Custom Functions
For more complex logic, define a regular Python function and apply it.
```py
def categorize_cost(total_cost):
    if total_cost < 200:
        return 'Low'
    elif 200 <= total_cost < 400:
        return 'Medium'
    else:
        return 'High'

# Create 'Cost_Category' column using a custom function
df_ops['Cost_Category'] = df_ops['Total_Cost'].apply(categorize_cost)
print("\nDataFrame with 'Cost_Category' column (using custom function):")
print(df_ops)

# Applying a function that uses multiple columns
def calculate_profit_margin(row):
    revenue = row['Total_Cost']
    cost_of_goods = row['Price'] * row['Quantity'] * 0.7 # Assume COGS is 70% of price * quantity
    profit = revenue - cost_of_goods
    return profit / revenue if revenue != 0 else 0

df_ops['Profit_Margin'] = df_ops.apply(calculate_profit_margin, axis=1)
print("\nDataFrame with 'Profit_Margin' column (using custom function on multiple columns):")
print(df_ops)
```
## 7. Applying Functions to Update Column Values
Sometimes you need to modify existing column values based on certain conditions or transformations.

### 7.1. Conditional Updates using loc or where
```py
data_update = {'Score': [75, 88, 62, 95, 50],
               'Grade': ['C', 'B', 'D', 'A', 'F']}
df_update = pd.DataFrame(data_update)

print("\nOriginal DataFrame for column update:")
print(df_update)

# Update 'Score' values: Add 5 points to scores below 60
df_update.loc[df_update['Score'] < 60, 'Score'] = df_update['Score'] + 5
print("\nDataFrame after updating 'Score' (add 5 to scores < 60):")
print(df_update)

# Update 'Grade' values: Change 'F' to 'Needs Improvement'
df_update.loc[df_update['Grade'] == 'F', 'Grade'] = 'Needs Improvement'
print("\nDataFrame after updating 'Grade' ('F' to 'Needs Improvement'):")
print(df_update)

# Using .where() for conditional replacement (replaces values where condition is FALSE)
df_update['Score_Adjusted'] = df_update['Score'].where(df_update['Score'] >= 70, 60) # Set scores below 70 to 60
print("\nDataFrame with 'Score_Adjusted' (using .where()):")
print(df_update)
```
### 7.2. Using map() for Value Replacement
map() is efficient for replacing values based on a dictionary or Series.
```py
# Create a mapping dictionary for grades
grade_mapping = {'A': 'Excellent', 'B': 'Good', 'C': 'Average', 'D': 'Pass', 'Needs Improvement': 'Fail'}

# Update 'Grade' column using map
df_update['Grade_Description'] = df_update['Grade'].map(grade_mapping)
print("\nDataFrame after updating 'Grade' with descriptions (using map):")
print(df_update)

7.3. Using replace() for Specific Value Replacements
# Replace specific values in 'Grade' column
df_update['Grade_Replaced'] = df_update['Grade'].replace({'Pass': 'Satisfactory', 'Average': 'Fair'})
print("\nDataFrame after replacing specific 'Grade' values (using replace):")
print(df_update)
```
### 7.4. Using apply() with Custom Functions for Complex Updates
```py
def adjust_score_based_on_grade(row):
    score = row['Score']
    grade = row['Grade']
    if grade == 'Excellent':
        return score + 2 # Bonus for excellent
    elif grade == 'Needs Improvement':
        return score - 5 # Penalty for needs improvement
    else:
        return score

# Create a new column 'Final_Score' based on conditional adjustments
df_update['Final_Score'] = df_update.apply(adjust_score_based_on_grade, axis=1)
print("\nDataFrame with 'Final_Score' (using apply with complex logic):")
print(df_update)
```
Conclusion
Data preprocessing is the bedrock of successful data science projects. Mastering these techniques will enable you to clean, transform, and prepare your data effectively, leading to more robust and accurate machine learning models. Always remember to explore your data thoroughly to understand its characteristics and choose the most appropriate preprocessing methods.
