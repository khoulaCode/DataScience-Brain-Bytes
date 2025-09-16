# Regression for Data Science



## Fundamentals of Regression and Linear Models

## Introduction to Regression

**What is Regression?**

Regression is a supervised machine learning task used to predict a continuous target variable based on one or more input features. Unlike classification, which predicts a categorical label (e.g., spam or not spam), regression predicts a numerical value (e.g., house price, temperature, sales).

**Model Evaluation Metrics**

Before we dive into the models, it's crucial to understand how to evaluate their performance. Here are the key metrics we'll use:

-   **Mean Absolute Error (MAE):** The average of the absolute differences between predicted and actual values. It's robust to outliers.
    
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

-   **Mean Squared Error (MSE):** The average of the squared differences. It penalizes larger errors more heavily than MAE.

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$  

-   **Root Mean Squared Error (RMSE):** The square root of the MSE. It's in the same units as the target variable, making it easier to interpret.

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

-   **R² (Coefficient of Determination):** Represents the proportion of variance in the dependent variable that is predictable from the independent variable(s). The best possible score is 1.0. A score of 0.0 indicates the model performs no better than a straight line at the mean of the data.

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

## Linear Regression

### Simple Linear Regression

This is the simplest form, using a single feature to predict a target.

**Mathematical Equation:** The goal is to find the best-fitting straight line that minimizes the sum of squared residuals.
$$y = \beta_0 + \beta_1 x + \epsilon$$
where:
-   `y` is the predicted value
-   `beta_0` is the y-intercept
-   `beta_1` is the slope
-   `x` is the input feature
-   `epsilon` is the error term

**Algorithm (Ordinary Least Squares):** The algorithm finds the values of `beta_0` and `beta_1` that minimize the cost function, which is the Mean Squared Error (MSE).

$$Cost(\beta_0, \beta_1) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1 x_i))^2$$

The analytical solution for the coefficients is given by the normal equation:

$$\hat{\beta} = (X^T X)^{-1} X^T y$$

**scikit-learn Example:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Generate some dummy data
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1) * 2

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R-squared: {r2_score(y_test, y_pred):.2f}")
```

**Hyperparameters:** Linear Regression has no significant hyperparameters to tune as it's a closed-form solution.

### Multiple Linear Regression

This is an extension of simple linear regression, using multiple features to predict the target.

**Mathematical Equation:**

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \epsilon$$

where:
- `x_1, x_2, ..., x_n` are the input features.

**scikit-learn Example:** The `LinearRegression` model handles multiple features automatically.
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Generate a dataset with multiple features
X, y = make_regression(n_samples=100, n_features=5, noise=15, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate
model = LinearRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"Model R-squared on test set: {score:.2f}")
```

### Regularization for Linear Models

Regularization techniques are used to prevent overfitting, especially with many features. They add a penalty to the cost function for larger coefficient values.

### Ridge Regression (L₂ Regularization)

Ridge regression adds a penalty equal to the square of the magnitude of the coefficients. This helps shrink coefficients towards zero but doesn't eliminate them.

**Cost Function:**

$$Cost(\beta) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} \beta_j^2$$

where `alpha` is the regularization strength.

**scikit-learn Example:**
```python
from sklearn.linear_model import Ridge

# Using the same X, y from Multiple Linear Regression example
ridge_model = Ridge(alpha=1.0) # alpha is the hyperparameter
ridge_model.fit(X_train, y_train)
score = ridge_model.score(X_test, y_test)
print(f"Ridge Regression R-squared: {score:.2f}")
```

**Hyperparameters:**
-   `alpha`: The regularization strength. A value of 0 is equivalent to standard Linear Regression. Larger values lead to more regularization.

### Lasso Regression (L₁ Regularization)

Lasso regression adds a penalty equal to the absolute value of the coefficients. It can shrink some coefficients all the way to zero, effectively performing feature selection.

**Cost Function:**

$$Cost(\beta) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} |\beta_j|$$

**scikit-learn Example:**
```python
from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
score = lasso_model.score(X_test, y_test)
print(f"Lasso Regression R-squared: {score:.2f}")
```

**Hyperparameters:**
-   `alpha`: The regularization strength. A larger alpha leads to more coefficients being shrunk to zero.

### Elastic Net

Elastic Net combines the penalties of both Ridge and Lasso, making it a good compromise.

**Cost Function:**

$$Cost(\beta) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \rho \sum_{j=1}^{p} |\beta_j| + \frac{\alpha(1-\rho)}{2} \sum_{j=1}^{p} \beta_j^2$$

where `rho` is the ratio of L₁ to L₂ regularization.

**scikit-learn Example:**
```python
from sklearn.linear_model import ElasticNet

elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5) # l1_ratio is rho
elastic_model.fit(X_train, y_train)
score = elastic_model.score(X_test, y_test)
print(f"ElasticNet R-squared: {score:.2f}")
```

**Hyperparameters:**
-   `alpha`: Overall regularization strength.
-   `l1_ratio`: The mixing parameter between Lasso and Ridge. A value of 0 is equivalent to Ridge, and 1 is equivalent to Lasso.

## Advanced and Non-Linear Regression

### Polynomial Regression

This technique models the relationship between the independent and dependent variables as an nth degree polynomial. It's an extension of linear regression that can capture non-linear relationships.

**Mathematical Equation:**

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_n x^n + \epsilon$$

**Algorithm:** It still uses Ordinary Least Squares after transforming the features into polynomial form.

**scikit-learn Example:** This requires using the `PolynomialFeatures` preprocessor.
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

# Generate non-linear dummy data
X = np.sort(np.random.rand(100, 1) * 20, axis=0)
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1) * 10

# Create a pipeline to handle the polynomial transformation and linear model
model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)), # Hyperparameter
    ('linear', LinearRegression())
])

model.fit(X, y)
score = model.score(X, y)
print(f"Polynomial Regression R-squared: {score:.2f}")
```

**Hyperparameters:**
-   `degree`: The degree of the polynomial features. A higher degree can lead to overfitting.

## Ensemble and Tree-Based Models

### Decision Tree Regression

A decision tree splits the data into branches based on feature values, creating a tree-like structure. The final leaf nodes contain the predicted value.

**Algorithm:** The algorithm recursively partitions the data by selecting the feature and split point that minimizes a cost function, such as MSE or MAE.

**scikit-learn Example:**
```python
from sklearn.tree import DecisionTreeRegressor

# Using the same X, y as Polynomial Regression example
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_model = DecisionTreeRegressor(max_depth=5) # Hyperparameter
tree_model.fit(X_train, y_train)
score = tree_model.score(X_test, y_test)
print(f"Decision Tree R-squared: {score:.2f}")
```

**Hyperparameters:**
-   `max_depth`: The maximum depth of the tree.
-   `min_samples_split`: The minimum number of samples required to split an internal node.
-   `min_samples_leaf`: The minimum number of samples required to be at a leaf node.

## Random Forest Regression

An ensemble method that builds multiple decision trees and averages their predictions to improve accuracy and control overfitting.

**Algorithm:** It builds a "forest" of trees, where each tree is trained on a random subset of the data and a random subset of features. The final prediction is the average of all individual tree predictions.

**scikit-learn Example:**
```python
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42) # Hyperparameters
forest_model.fit(X_train, y_train.ravel())
score = forest_model.score(X_test, y_test)
print(f"Random Forest R-squared: {score:.2f}")
```

**Hyperparameters:**
-   `n_estimators`: The number of trees in the forest.
-   `max_depth`: Maximum depth of each tree.
-   `min_samples_split`, `min_samples_leaf`: Same as Decision Trees.

### Gradient Boosting Regression

Another powerful ensemble method that builds trees sequentially. Each new tree corrects the errors of the previous ones.

**Algorithm:** It starts with an initial prediction, then iteratively builds new "weak learner" trees (usually shallow decision trees) to correct the residual errors. It then adds these new trees' predictions to the existing model.

**scikit-learn Example:**
```python
from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42) # Hyperparameters
gb_model.fit(X_train, y_train.ravel())
score = gb_model.score(X_test, y_test)
print(f"Gradient Boosting R-squared: {score:.2f}")
```

**Hyperparameters:**
-   `n_estimators`: The number of boosting stages (trees).
-   `learning_rate`: Controls the contribution of each tree.
-   `max_depth`: The maximum depth of the individual trees.

### Support Vector Regression (SVR)

SVR is a powerful algorithm that finds the best fit line (or hyperplane) for the data while tolerating a certain amount of error (epsilon) on each side of the line.

**Algorithm:** SVR uses a technique called the "kernel trick" to handle non-linear data. It aims to find a function that has at most epsilon deviation from the training data while being as flat as possible. The flatness is controlled by the C-parameter.

**scikit-learn Example:**
```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Scale the data first, as SVR is sensitive to feature scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Initialize and train SVR model
svr_model = SVR(kernel='rbf', C=10, epsilon=0.1) # Hyperparameters
svr_model.fit(X_train_sc, y_train_sc.ravel())

# Make predictions and inverse transform to original scale
y_pred_sc = svr_model.predict(X_test_sc)
y_pred_orig = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1))
y_test_orig = scaler_y.inverse_transform(y_test_sc.reshape(-1, 1))

print(f"SVR R-squared: {r2_score(y_test_orig, y_pred_orig):.2f}")
```

**Hyperparameters:**
-   `kernel`: The kernel function to use ('linear', 'poly', 'rbf', 'sigmoid'). The 'rbf' (Radial Basis Function) is a popular choice for non-linear problems.
-   `C`: The regularization parameter. Smaller values of C mean a larger regularization and a wider margin.
-   `epsilon`: The maximum error margin allowed for the model.
