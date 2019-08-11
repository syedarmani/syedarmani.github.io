---
title: 'NumPy Fundamentals'
date: 2019-08-11
permalink: /posts/2019/08/numpy-fundamentals/
tags:
  - Artificial Intelligence
  - Python
---

NumPy stands for _Numerical Python_. It provides fast mathematical computation on arrays and matrices. In NumPy, dimensions are called axes. The number of axes is called the rank. In this post we will go through some of the basic functionality available with NumPy.

The easiest way to create an array is to use the array function: 
```python 
import numpy as np
first_arr = np.array([0, 0.5, 1.0, 1.5, 2.0])
print(type(first_arr))
<class numpy.ndarray>

```

Generate evenly spaced values within a given interval:
```python
#numpy.arange(start,stop,step,dtype=None)
second_arr = np.arange(2, 20, 2)
print(second_arr)

[ 2  4  6  8 10 12 14 16 18]

```

Calculate sum of all elements in _third_arr_ array:
```python
third_arr = np.arange(8, dtype=np.float)
print(third_arr)
print(third_arr.sum())

[0. 1. 2. 3. 4. 5. 6. 7.]
28.0
```

Calculate _standard deviation_ of elements:
```python
print(third_arr.std())
2.29128784747792
```

Mathematical operations on ndarray objects:
```python
print(2 * third_arr)
[ 0.  2.  4.  6.  8. 10. 12. 14.]

print(third_arr ** 2)
[ 0.  1.  4.  9. 16. 25. 36. 49.]

print(2 ** third_arr)
[  1.   2.   4.   8.  16.  32.  64. 128.]

```
Create a 2D array:
```python
twoD_arr = np.array([third_arr, third_arr * 2])
print(twoD_arr)
[[ 0.  1.  2.  3.  4.  5.  6.  7.]
 [ 0.  2.  4.  6.  8. 10. 12. 14.]]
```

Add the elements of ndarray column-wise:
```python
print(twoD_arr.sum(axis=0))
[ 0.  3.  6.  9. 12. 15. 18. 21.]
```

Add the elements of ndarray row-wise:
```python
print(twoD_arr.sum(axis=1))
[28. 56.]
```
Initialize an array populated with zeros:
```python
zeros_arr = np.zeros((2, 3))
print(zeros_arr)
```
Initialize an array populated with ones:
```python
ones_arr = np.ones((2, 3))
print(ones_arr)
```
Square matrix with diagonal populated with ones:
```python
print(np.eye(3))

array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
```
Create a one-dimensional ndarray with evenly spaced numbers over a specified interval:
```python
lin_arr = np.linspace(5,10,5)
print(lin_arr)
[ 5.    6.25  7.5   8.75 10.  ]
```
Some attributes of ndarray object:
```python
print(lin_arr.size)
5

print(lin_arr.itemsize)
8

print(lin_arr.ndim)
1

print(lin_arr.shape)
(5,)

print(lin_arr.dtype)
float64

```

Reshaping a numpy array:
```python
arr_sh = np.arange(6)
print(arr_sh)
[0 1 2 3 4 5]

print(arr_sh.shape)
(6,)

print(arr_sh.reshape(2,3))
[[0 1 2]
 [3 4 5]]

print(arr_sh.reshape(3,2))
[[0 1]
 [2 3]
 [4 5]]

```

Resizing a numpy array:
```python
arr_re = np.arange(12)
print(arr_re)
[ 0  1  2  3  4  5  6  7  8  9 10 11]

print(np.resize(arr_re, (3,1)))
[[0]
 [1]
 [2]]

h_arr = np.resize(arr_re, (4,3))
print(h_arr)
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]

```
Horizontal Stacking:
```python
print(np.hstack((h_arr, 2*h_arr)))
[[ 0  1  2  0  2  4]
 [ 3  4  5  6  8 10]
 [ 6  7  8 12 14 16]
 [ 9 10 11 18 20 22]]

```

Vertical Stacking:
```python
print(np.vstack((h_arr, 2*h_arr)))

[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]
 [ 0  2  4]
 [ 6  8 10]
 [12 14 16]
 [18 20 22]]
```

Flatten a numpy array row-wise(default):
```python
print(h_arr.flatten(order='C'))

[ 0  1  2  3  4  5  6  7  8  9 10 11]
```

Flatten a numpy array column-wise:
```python
print(h_arr.flatten(order='F'))
[ 0  3  6  9  1  4  7 10  2  5  8 11]
```
Boolean arrays:
```python
print(h_arr > 5)

[[False False False]
 [False False False]
 [ True  True  True]
 [ True  True  True]]

print((h_arr > 1) & (h_arr < 5))

[[False False  True]
 [ True  True False]
 [False False False]
 [False False False]]

print(h_arr[(h_arr > 1) & (h_arr < 5)])
[2 3 4]
```

Use *np.where* to return elements depending on condition:
```python
np.where(h_arr < 5, 1,0)
array([[1, 1, 1],
       [1, 1, 0],
       [0, 0, 0],
       [0, 0, 0]])

np.where(h_arr % 2 == 0, 'even', 'odd')
array([['even', 'odd', 'even'],
       ['odd', 'even', 'odd'],
       ['even', 'odd', 'even'],
       ['odd', 'even', 'odd']], dtype='<U4')
```

