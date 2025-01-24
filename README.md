# Simple Linear Regression
## Introduction

Linear regression is a statistical method for modeling the relationship between a dependent variable and one or more independent variables. It is widely used in data analysis and machine learning.

## Prerequisites

- Basic understanding of Python
- Familiarity with Jupyter Notebooks
- Knowledge of basic statistics

## Installation

To install the necessary packages, run the following command:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Usage

1. Import the required libraries:

    ```python
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import matplotlib.pyplot as plt # data visualization
    from sklearn import linear_model # linear regression
    ```

2. Load your dataset:

    ```python
data = pd.read_csv('canada_per_capita_income.csv') # load data set
data.head(2) # show first 2 rows of data set
    ```

3. Prepare your data:

    ```python
    X = data[['independent_variable']]
    y = data['dependent_variable']
    ```

4. Create and train the model:

    ```python
reg = linear_model.LinearRegression() # create linear regression object
reg.fit(data[['year']], data.income)
    ```

5. Make predictions:

    ```python
reg.predict([[2020]]) # predict income in 2020
reg.coef_ # slope of line
reg.intercept_ # intercept of line
    ```

6. Visualize the results:

    ```python
%matplotlib inline
plt.scatter(data.year, data.income, color='green', marker='*') # scatter plot
plt.plot(data.year, reg.predict(data[['year']]), color='red') # line plot
plt.xlabel('Year') # x-axis label
plt.ylabel('Income') # y-axis label
plt.title('Income per year') # title of plot
plt.show() # show plot
    ```

7.Predict Model Accuracy:
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predict the values for the test set
y_pred = reg.predict(data[['year']])

# Calculate MAE
mae = mean_absolute_error(data['income'], y_pred)
print(f'Mean Absolute Error (MAE): {mae}')

# Calculate MSE
mse = mean_squared_error(data['income'], y_pred)
print(f'Mean Squared Error (MSE): {mse}')

# Calculate R-squared
r2 = r2_score(data['income'], y_pred)
print(f'R-squared: {r2}')

```

## Conclusion

Linear regression is a powerful tool for predictive analysis. By following the steps outlined in this guide, you can implement linear regression in your own projects.