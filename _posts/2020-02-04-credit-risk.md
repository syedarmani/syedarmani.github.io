---
title: 'Credit Risk Analysis with Machine Learning, Part 1.'
date: 2020-02-04
permalink: /posts/2020/04/credit-risk/
tags:
  - Finance
  - Python
  - Artificial Intelligence 
---

In the Part 1 of this series on credit risk modelling, we will use logistic regression to create a model which will predict the probability of default by the borrowers. In Part 2, we will explore random forrest algorithm. Let's begin by importing all the required libraries:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_curve, classification_report, confusion_matrix
```

Load the dataset to a Pandas DataFrame.
```python
df = pd.read_csv("credit.csv")
```

Let's have a look at the first 5 rows of our dataset.
```python
df.head()
```
![png](/images/plots/crm1.png )

Some other basic information about the dataset.
```python
df.shape

Output: 
(32048, 9)
```
```python
df.info()

Output:

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 32048 entries, 0 to 32047
Data columns (total 9 columns):
age                    32048 non-null int64
income                 32048 non-null int64
home_ownership         32048 non-null object
job_length             31159 non-null float64
loan_reason            32048 non-null object
amount                 32048 non-null int64
interest_rate          28989 non-null float64
loan_status            32048 non-null int64
loan_percent_income    32048 non-null float64
dtypes: float64(3), int64(4), object(2)
```

The **loan_status** column gives information about the default by the borrowers, the value of 1 is default and 0 for non default.

```python
print(df['loan_status'].head())

Output:

0    0
1    0
2    1
3    1
4    0
Name: loan_status, dtype: int64

```

Let's create a scatter plot for 'age' of the person and the 'amount' borrowed.
```python
plt.scatter(df['age'], df['amount'], c='red', alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Amount")
plt.show()
```
![png](/images/plots/crm2.png )

Let us see which columns have missing values:
```python
print(df.columns[df.isnull().any()])

Output:
Index(['job_length', 'interest_rate'], dtype='object')
```

Now, let's look at the missing values in the interest rate column:
```python
display(df[df['interest_rate'].isnull()].head())
```
![png](/images/plots/crm3.png )

Replace the missing values with the median value for all interest rates.
```python
df['interest_rate'].fillna((df['interest_rate'].median()), inplace=True)
```
Remove missing data.
```python
# Print the number of missing values in the job_length column.
display(df['job_length'].isnull().sum())

Output:
889
```
```python
# Get the indices
ind = df[df['job_length'].isnull()].index

# Save the new dataset without missing values.
df_no_missing_data = df.drop(ind)

print(df_no_missing_data.columns[df_no_missing_data.isnull().any()])

Output:

Index([], dtype='object')
```

One hot encoding.
```python
df_num = df_no_missing_data.select_dtypes(exclude=['object'])
df_str = df_no_missing_data.select_dtypes(include=['object'])
df_str_onehot = pd.get_dummies(df_str)
df_loan_prep = pd.concat([df_num, df_str_onehot], axis=1)
```
## Building a Logistic Regression Model

We turn now to construct a logistic regression classifier. The form of the hypothesis $h_\theta(x)$ for logistic regression is:

$$h_\theta(x) = g(\theta^Tx)$$
  
where the function $g$ is the sigmoid function:

$$g(z) = \frac{1}{1+e^{-z}}$$

### Cost Function J$(\theta)$ : The cost function for logistic regression is

$$J(\theta) = \frac{1}{m}\sum_{i=1}^{m} \left(-y^{(i)} \log\left[h_{\theta}(x^{(i)})\right] - (1- y^{(i)}) \log\left[1-h_{\theta}(x^{(i)})\right]\right)$$

which can be rewritten as

$$J(\theta) = - \frac{1}{m} \left(\sum_{i=1}^{m} y^{(i)} \log\left[  h_{\theta}(x^{(i)})\right] + (1- y^{(i)}) \log\left[1-h_{\theta}(x^{(i)})\right] \right)$$

The gradient of this cost function is a vector which is the same length $n$ as $\theta$, where the $j$th element (for $j = 0,1,2,\ldots,n)$ is

$$ \frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i = 1}^{m} \left(h_{\theta}(x^{(i)}) - y^{(i)}  \right) x^{(i)}_j$$
 

Now, we will use the `LogisticRegression` function from sklearn.

```python
X = df_loan_prep.copy()
y = df_loan_prep[['loan_status']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

lr_classifier = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))
predictions = lr_classifier.predict_proba(X_test)
predictions_df = pd.DataFrame(predictions[:,1][0:5], columns = ['default_probability'])
real_df = y_test.head()
preds_df = pd.DataFrame(preds[:,1], columns = ['default_probability'])
preds_df['loan_status'] = preds_df['default_probability'].apply(lambda x: 1 if x > 0.50 else 0)

# Print the classification report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, preds_df['loan_status'], target_names=target_names))

Output:

Name: loan_status, dtype: int64
              precision    recall  f1-score   support

 Non-Default       0.78      0.95      0.86      9753
     Default       0.24      0.06      0.10      2711

   micro avg       0.76      0.76      0.76     12464
   macro avg       0.51      0.50      0.48     12464
weighted avg       0.67      0.76      0.69     12464

```

Let's check out the accuracy of the model.
```python
print(lr_classifier.score(X_test, y_test))

Output:

0.8080070603337612

```
# Plot the ROC curve.
```python
prob_default = predictions[:, 1]
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, prob_default)
plt.plot(false_positive_rate, true_positive_rate, color = 'red')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.show()

```
![png](/images/plots/crm4.png )

Confusion matrix.
```python
tn, fp, fn, tp = confusion_matrix(y_test,predictions_df['loan_status']).ravel()
print(" Total Negative", tn, "\n False Positive", fp, "\n False Negative", fn, "\n Total Positive", tp)

Output:

 Total Negative 9652 
 False Positive 174 
 False Negative 2219 
 Total Positive 419

```
