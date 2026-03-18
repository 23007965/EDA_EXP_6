```
Name : P PARTHIBAN
Reg No. : 212223230145
```
# EDA_EXP_6

## Aim

To perform complete Exploratory Data Analysis (EDA) on the Wine Quality dataset, detect and remove outliers using the IQR method, and compare the performance of a classification model (Logistic Regression) before and after outlier removal.

## Algorithm

1)Import pandas, numpy, seaborn, matplotlib, sklearn libraries.

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


**Output**

**Result**

