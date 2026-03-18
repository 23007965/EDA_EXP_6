```
Name : P PARTHIBAN
Reg No. : 212223230145
```
# EDA_EXP_6

## Aim

To perform complete Exploratory Data Analysis (EDA) on the Wine Quality dataset, detect and remove outliers using the IQR method, and compare the performance of a classification model (Logistic Regression) before and after outlier removal.

## Algorithm

Step 1 - Load the wine dataset and check its basic structure, shape, and missing values.

Step 2 - Plot univariate distributions for alcohol, volatile acidity, and pH to understand individual feature behavior.

Step 3 - Create bivariate boxplots to study relationships between wine quality and key predictors.

Step 4 - Compute and visualize the correlation matrix to identify feature relationships with wine quality.

Step 5 - Convert wine quality into a binary good/bad label for classification.

Step 6 - Split the dataset into training and testing sets for model evaluation.

Step 7 - Train a Logistic Regression model and predict wine quality on test data.

Step 8 - Evaluate the model using accuracy and confusion matrix, and detect outliers using boxplots.


## Program

### Your Name : P PARTHIBAN

### Your Reg No. : 212223230145

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')
print(df.head())
print(df.shape)
print(df.dtypes)
```
```python
features = ['alcohol','volatile acidity','pH']
summary = df[features].agg(['mean','median','std'])
print(summary)
```
```python
plt.figure(figsize=(9,3))
for i, col in enumerate(features,1):
    plt.subplot(1,3,i)
    sns.histplot(df[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()
```
```python
plt.figure(figsize=(8,3))
plt.subplot(1,2,1)
sns.boxplot(x='quality', y='alcohol', data=df)
plt.subplot(1,2,2)
sns.boxplot(x='quality', y='volatile acidity', data=df)
plt.show()
```
```python
plt.figure(figsize=(8,3))
plt.subplot(1,2,1)
sns.scatterplot(x='alcohol', y='quality', data=df)
plt.subplot(1,2,2)
sns.scatterplot(x='volatile acidity', y='quality', data=df)
plt.show()
```
```python
numeric_cols = df.select_dtypes(include=np.number).columns
outlier_counts = {}
for col in numeric_cols:
  Q1 = df[col].quantile(0.25)
  Q3 = df[col].quantile(0.75)
  IQR = Q3 - Q1
  lower = Q1 - 1.5*IQR
  upper = Q3 + 1.5*IQR
  outliers = df[(df[col] < lower) | (df[col] > upper)]
  outlier_counts[col] = len(outliers)
print(outlier_counts)
```
```python
sns.regplot(x='alcohol', y='quality', data=df)
plt.show()
```
```python
corr = df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, cmap='coolwarm')
plt.show()
print(corr['quality'].sort_values(ascending=False))
```


## Output

<img width="659" height="597" alt="image" src="https://github.com/user-attachments/assets/5000f2ca-59c3-4b4b-96eb-e8969ff998b5" />

<img width="918" height="289" alt="image" src="https://github.com/user-attachments/assets/0514c2fa-80df-4865-8f90-785859512a51" />

<img width="728" height="288" alt="image" src="https://github.com/user-attachments/assets/2055dbca-fc58-4e3d-9b87-8ad9d83f02fe" />

<img width="720" height="296" alt="image" src="https://github.com/user-attachments/assets/da701fbe-33c4-4eea-95db-1866e0e76324" />

<img width="1350" height="19" alt="image" src="https://github.com/user-attachments/assets/12fc97cd-cc37-4e6a-83cf-1decc3af372a" />

<img width="630" height="441" alt="image" src="https://github.com/user-attachments/assets/257db87f-2346-4320-9079-5542ca9844aa" />

<img width="859" height="623" alt="image" src="https://github.com/user-attachments/assets/0d260402-65d0-4fd6-a1bd-7c7afa51dbfd" />

<img width="540" height="239" alt="image" src="https://github.com/user-attachments/assets/070498bc-af38-4c17-8e6f-bd37edf147b2" />

## Result
Thus, To perform complete Exploratory Data Analysis (EDA) on the Wine Quality dataset, detect and remove outliers using the IQR method, and compare the performance of a classification model (Logistic Regression) before and after outlier removal has successfully completed.
