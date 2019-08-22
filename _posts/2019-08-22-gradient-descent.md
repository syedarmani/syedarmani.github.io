---
title: 'Gradient Descent'
date: 2019-08-22
permalink: /posts/2019/08/gradient-descent/
tags:
  - Artificial Intelligence
  - Python
---

Gradient descent, also known as steepest descent is an optimization algorithm for finding the local minimum of a function. 

Gradient (▽C(x,y)) of a function (C(x,y)) gives you the direction of steepest ascent, basically which direction should you step to increase the function most quickly. So, taking the negative of that gradient gives you the direction that decreases the function most quickly. The length of this gradient vector is actually an indication for just just how steep that steepest slope is.

Algorithm:

* Compute ▽C
* Small step in -▽C
* Repeat



In machine learning, we use gradient descent to update the parameters of our model. Parameters refer to coefficients in Linear Regression and weights in neural networks. Neural network learning is just the process of minimizing the cost function.

Now we will implement gradient descent to fit the linear regression parameters $\theta = (\theta_0,\theta_1)$ to the data set, `data`. 
The adjustable parameters of this regression model are $\theta_0$ and $\theta_1$, which control the $y$-intercept and slope, 
respectively. These are the values that gradient descent changes to minimize the cost $J(\boldsymbol{\theta})$.

<br />
![png](/images/plots/gd.png)
<center><sub>Image source: http://blog.datumbox.com/wp-content/uploads/2013/10/gradient-descent.png </sub></center>
<br />

For univariate linear regression, the structure of the hypothesis is very simple: It is the equation for the slope of a line.  This equation specifies how 
$\theta$ and $x$ are to be used to compute an estimate of the dependent variable, $y$. Specifically, the hypothesis 
$h_\theta(x)$ is determined by $\theta$, a two-dimensionpal vector $\theta = (\theta_0,\theta_1)$, and the single feature $ x_1$, such that: 
<center>$h_\theta(x) = \theta_0 + \theta_1x_1.$</center><br />

To fit a model of this form to data, we need a way to pick values for $\theta_0$ and $ \theta_1$, that "best fit" the data. We want to select parameters that minimize the cost function. Cost function $J(\boldsymbol{\theta})$ is given by this formula:

<center> $J(\boldsymbol{\theta}) = \frac{1}{2m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$</center>
<br /> 

**m**, is the number of training examples, the objective here is to minimize the cost function $J(\boldsymbol\theta)$, written as $\min_{\boldsymbol{\theta}} J(\boldsymbol\theta)$, which is achieved by selecting values for $\boldsymbol{\theta}$ that makes the sum of squared errors as small as possible. 
The fraction $\frac{1}{m}$ is included to normalize the score. There are a family of gradient descent algorithms. We will implement **batch gradient descent**.
<center>$\theta_j := \theta_j - \alpha \frac{1}{2} \sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})x^{(i)}$ </center>

<br />

```python

# Implementation of batch gradient descent.
import numpy as np
import matplotlib.pyplot as plt

#Load the dataset
data = np.loadtxt('../data/ps1_data.csv', delimiter=',')

#Number of rows
m = np.shape(data)[0]

# initialize 'X' and 'y' array
X = np.ones((m, 2))
X[:,1] = data[:,0]
y = np.array(data[:,1])


# initalize model
theta = np.zeros(2)

def costJ(X, y, theta):
    """Implement the squared-error cost function, costJ(X, y, theta), which computes the cost of using the
    theta to parameterize a linear regression hypothesis to fit the data.
    """
    scores = np.power((np.dot(X,theta.T) - y), 2)
    cost = np.sum(scores) / (2 * len(X))
    return cost

def batchGradientDescent(X, y, theta, alpha, num_iterations):
    """batchGradientDescent performs batch gradient descent to minimize theta and return a history
    of the gradient steps to plot.

    It updates theta by taking a number of gradient steps, fixed by the
    parameter num_iterations, where the size of those steps is determined by the learning
    rate, alpha.
    """

    # m is the number of training samples
    m= len(y)

    # J_history is the cost function history, intialized to a vector of zeros
    # for each iteration step
    J_history = np.zeros(num_iterations)
    
    # initialize theta_new with a copy of theta.
    theta_new = theta.copy()
    for i in range(num_iterations):
        #update theta_new
        theta_new -= (alpha/m) * (X.T.dot(np.dot(X, theta_new)-y))
        
        # Update J_history with compute_cost using theta_new
        J_history[i] = costJ(X, y, theta_new)
                                  
                                  
    return theta_new, J_history

# learning rate
alpha = 0.01

#number of iterations
iterations = 1500

# run gradient descent
theta_new, J_history = batchGradientDescent(X, y, theta, alpha, iterations)

```

```python
print(theta_new)

Output: 
[8.83961111 1.15819095]
```
Plot the history of cost function:

```python
plt.plot(J_history)
plt.ylabel(r"$J(\theta)$")
plt.xlabel("Iteration number")
```
<br />
![png](/images/plots/cost.png)

**Learning Rate($\alpha$)**: The size of these steps is called the learning rate. With a high learning rate we can cover more ground each step, but we risk overshooting. With a very low learning rate, we can confidently move in the direction of the negative gradient but calculating the gradient is time-consuming, so it will take us a very long time to get to the bottom.
