# Employee Machine Learning Project

## Overview
This project demonstrates a complete Machine Learning workflow using an employee dataset. It covers data preprocessing, feature engineering, feature selection, model training, and prediction using Linear Regression.

## Dataset
The dataset contains employee information such as:
- Education
- Joining Year
- City
- Payment Tier
- Age
- Gender
- Ever Benched
- Experience in Current Domain
- Leave Or Not

## Project Workflow

1. Load the employee dataset
2. Handle missing values using SimpleImputer
3. Apply Log Transformation to reduce skewness
4. Create a High Cardinality feature (Employee_ID)
5. Apply Target Encoding
6. Perform Feature Selection using SelectKBest
7. Split the dataset into training and testing sets
8. Train a Linear Regression model
9. Predict employee age
10. Compare predicted values with actual values

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Category Encoders

## Libraries

```python
pandas
numpy
scikit-learn
category_encoders
