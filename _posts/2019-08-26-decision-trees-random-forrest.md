---
title: 'Decision Trees and Random Forrest'
date: 2019-08-26
permalink: /posts/2019/08/random-forrest/
tags:
  - Artificial Intelligence
  - Python
---

Decision Trees are a type of ML algorithms which can be used for both Classificaton and Regression. They are also the core component of Random Forrests, which is one of the most powerful ML algorithms. Let us load the "iris" dataset provided by scikit-learn, train a classifier on it and then visualize it. Scikit-Learn uses the CART algorithm, which produces only binary trees:

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

iris = load_iris()

# petal length and width
X = iris.data[:, :2] 
y = iris.target


tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

Output:

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')

```


```python
print(iris.feature_names[:2])
print(iris.target_names)

export_graphviz(
        tree_clf,
        out_file="iris_tree.dot",
        feature_names=iris.feature_names[:2],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )
! dot -Tpng "iris_tree.dot" -o tree_2.png

Output:

['sepal length (cm)', 'sepal width (cm)']
['setosa' 'versicolor' 'virginica']
```

When max_depth is set to 2:
![png](/images/plots/tree_2.png)

At the root node(depth=0), it checks whether the sepal length is less than equal to 5.45, if yes then we move down to the left child node which checks another condition i.e Is sepal width <= 2.8? if that is true then we move down to the left child node again, which is a leaf node that tells us about the predicted class. In this case it is "versicolor".

**Gini Impurity**: If we have _N_ total classes and p(i) is the probability of picking a datapoint with class i, then the Gini Impurity is calculated as

\begin{align} G =  \sum_{i=1}^{N}  p(i) * (1 - p(i))\end{align}

In other words, it is the probability of incorrectly classifying a randomly chosen element in the dataset.

**Entropy**: A set's entropy is zero when all its elements are of the same class. 
\begin{align} E =  \sum_{i=1}^{N}  - p_i * log_2(p_i)\end{align}

**Classification And Regression Tree (CART)**: For **Classification** tasks, it first splits the training set in two subsets using a single feature and a threshold. It keeps on splitting until max_depth is reached or it cannot find a split that could reduce the impurity further.
Cost function for classification:

\begin{align} J =  \frac{m_l} {m} * G_l + \frac{m_r} {m} * G_r \end{align}
where,

m = total number of instances 

$m_l$= number of instances in left subset

$m_r$= number of instances in right subset

$G_l$= Impurity of left subset

$G_r$= Impurity of right subset


For **Regression** tasks, it splits the training set to minimize the MSE. Cost function for regression:

\begin{align} J =  \frac{m_l} {m} * MSE_l + \frac{m_r} {m} * MSE_r \end{align}

# Random Forrests
A **Random Forrests** is an _ensemble_ of "Decision Trees", while ensemble refers to a group of predictors. The idea behind random forrests is that the average of the predictions of a group of predictors gives better predictions than with the best individual predictor. Let's create a Random Forrest Classifier for MNIST dataset.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

```

```python
from sklearn.datasets import load_digits
digits = load_digits()
digits.keys()

Output: 

dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])

```

```python
fig = plt.figure(figsize= (5,5))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(64):
    plt.subplot(8,8,i+1)
    plt.imshow(digits.images[i], cmap = plt.cm.binary, interpolation='nearest')
    plt.text(0,0, digits.target[i], horizontalalignment='left',verticalalignment='top')
    plt.xticks([]),plt.yticks([])


```
![png](/images/plots/mnist_rf.png )

```python
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state = 10)

# build a forest of 10000 estimators with maximum tree depth of 4
rfc = RandomForestClassifier(max_depth = 4, n_estimators=1000,random_state = 10)

# fit rfc to your training data
fit_rfc= rfc.fit(Xtrain, ytrain)

# test on your test set
ypred = rfc.predict(Xtest)

#Check f1 score.
print(metrics.classification_report(ypred, ytest))


```
Output:

```
              precision    recall  f1-score   support

           0       0.98      0.98      0.98        46
           1       0.84      0.90      0.87        41
           2       0.96      0.84      0.90        51
           3       0.92      0.96      0.94        46
           4       0.93      0.93      0.93        40
           5       0.95      0.89      0.92        44
           6       1.00      1.00      1.00        47
           7       0.96      0.88      0.92        51
           8       0.86      0.90      0.88        41
           9       0.82      0.93      0.87        43

   micro avg       0.92      0.92      0.92       450
   macro avg       0.92      0.92      0.92       450
weighted avg       0.92      0.92      0.92       450

```

