---
title: 'Data Analysis with Pandas'
date: 2019-08-12
permalink: /posts/2019/08/pandas/
tags:
  - Artificial Intelligence
  - Python
---

Import the pandas and NumPy libraries:
```python
import pandas as pd
import numpy as np
```

A pandas series is a one dimentional labeled array. Create a pandas series:
```python
pd_ser = pd.Series([1,2,3], index=['a','b','c'])
print(pd_ser)

Output:

a    1
b    2
c    3
dtype: int64

```

A pandas DataFrame is a two dimensional labeled data structure with columns having data of different data types. Create a pandas DataFrame:

```python
data = {"str_col": ['a', 'b', 'c'],
     "int_col": [5,2,3], 
     "str_col": ['a', 'b', 'c'], 
     "float_col": ['0.1', '0.5', '0.10'], 
     "bool_col": ['True', 'False', 'True']
    }
df = pd.DataFrame(data, columns=['str_col', 'int_col', 'float_col', 'bool_col'], index=[1,2,3])
print(df)

Output:

  str_col  int_col float_col bool_col
1       a        5       0.1     True
2       b        2       0.5    False
3       c        3      0.10     True
```

Getting more information with the help() method:
```python
help(pd.DataFrame.sort_index)

Output:
Help on function sort_index in module pandas.core.frame:
sort_index(self, axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, by=None)
```

Print the first element of the series:
```python
print(pd_ser['a'])

Output:
1
```

Print the first entry of the DataFrame:
```python
print(df[0:1])

Output:

  str_col  int_col float_col bool_col
1       a        5       0.1     True
```

Print the the _str_col_ column of the dataframe:
```python
print(df['str_col'])
Output:
1    a
2    b
3    c
Name: str_col, dtype: object
```

Print a value based on its position with df.iloc[[row],[col]]
```python
print(df.iloc[[0],[0]])
Output:
  str_col
1       a
```

Select an entry based on row and label name:
```python
print(df.loc[[1],['str_col']])
Output:
  str_col
1       a
```

Get all rows where value of 'int_col' is greater than 1:
```python
df[df['int_col'] > 1]
Output:
  str_col  int_col float_col bool_col
1       a        5       0.1     True
2       b        2       0.5    False
3       c        3      0.10     True


```

Get all rows where value of bool_col is 'True':
```python
print(df[df['int_col'] > 1])
Output:

  str_col  int_col float_col bool_col
1       a        5       0.1     True
2       b        2       0.5    False
3       c        3      0.10     True
```

Add a new column:
```python
df['new_col'] = ['A0', 'B1', 'C3']
print(df)

Output:
  str_col  int_col float_col bool_col new_col
1       a        5       0.1     True      A0
2       b        2       0.5    False      B1
3       c        3      0.10     True      C3


```
Change the value of an element in the 'new_col':
```python
df.at[2,'new_col'] = 'C2'
print(df)

Output:

  str_col  int_col float_col bool_col new_col
1       a        5       0.1     True      A0
2       b        2       0.5    False      B1
3       c        3      0.10     True      C2

```
Drop the 'new_col' column:
```python
df.drop('new_col', axis=1, inplace=True)
print(df)

Output:
  str_col  int_col float_col bool_col
1       a        5       0.1     True
2       b        2       0.5    False
3       c        3      0.10     True

```
Create a new DataFrame _df2_ :
```python
data = {"col_1": [12,9,56,56],
     "col_2": [5,2,3,3], 
     "col_3": [16,18,13,13], 
     "col_4": [0.1, 0.5, 0.10,0.10], 
    }
df2 = pd.DataFrame(data, columns=['col_1', 'col_2', 'col_3', 'col_4'], index=[10,20,3,4])
print(df2)
Output:

    col_1  col_2  col_3  col_4
10     12      5     16    0.1
20      9      2     18    0.5
3      56      3     13    0.1
4      56      3     13    0.1
```

Sort the values in the 'int_col' column:
```python
print(df2.sort_values('col_3'))
Output:

    col_1  col_2  col_3  col_4
3      56      3     13    0.1
4      56      3     13    0.1
10     12      5     16    0.1
20      9      2     18    0.5

```

Sort the index:
```python
print(df2.sort_index())
Output:

    col_1  col_2  col_3  col_4
3      56      3     13    0.1
4      56      3     13    0.1
10     12      5     16    0.1
20      9      2     18    0.5

```

Print sum of columns:
```python
print(df2.sum())
Output:

col_1    133.0
col_2     13.0
col_3     60.0
col_4      0.8
dtype: float64
```
Print the minimum value:
```python
print(df2.min())
Output:

col_1     9.0
col_2     2.0
col_3    13.0
col_4     0.1
dtype: float64

```

Print the maximum value:
```python
print(df2.max())
Output:

col_1    56.0
col_2     5.0
col_3    18.0
col_4     0.5
dtype: float64
```

Describe the DataFrame:
```python
print(df2.describe())
Output:

           col_1     col_2     col_3  col_4
count   4.000000  4.000000   4.00000    4.0
mean   33.250000  3.250000  15.00000    0.2
std    26.297972  1.258306   2.44949    0.2
min     9.000000  2.000000  13.00000    0.1
25%    11.250000  2.750000  13.00000    0.1
50%    34.000000  3.000000  14.50000    0.1
75%    56.000000  3.500000  16.50000    0.2
max    56.000000  5.000000  18.00000    0.5

```

Print the mean:
```python
print(df2.mean())

Output:

col_1    33.25
col_2     3.25
col_3    15.00
col_4     0.20
dtype: float64

```

Print the median:
```python
print(df2.median())
Output:

col_1    34.0
col_2     3.0
col_3    14.5
col_4     0.1
dtype: float64

```
Apply a function to the DataFrame:
```python
k = lambda x: x* 10
print(df2.apply(k))
Output:

    col_1  col_2  col_3  col_4
10    120     50    160    1.0
20     90     20    180    5.0
4     560     30    130    1.0

```

GroupBy Operations:
```python
data = {"col_1": [1,2,3,4,5,6,7,8,9],
     "col_2": [10,11,12,13,14,15,16,17,18], 
     "col_3": [20,21,22,23,24,25,26,27,28],
     "col_4": [100,200,300,100,200,300,100,200,300],
        
    }
third_df= pd.DataFrame(data, columns=['col_1', 'col_2', 'col_3', 'col_4'],index=[np.arange(9)])
groups = third_df.groupby('col_4')
groups.size()

Output:
col_4
100    3
200    3
300    3
dtype: int64

```

Print the mean of groups:
```python
print(groups.mean())
Output:

       col_1  col_2  col_3
col_4                     
100        4     13     23
200        5     14     24
300        6     15     25

```

Print the maximum value in the groups:
```python
print(groups.max())

Output:
       col_1  col_2  col_3
col_4                     
100        7     16     26
200        8     17     27
300        9     18     28
```

Print aggregates:
```python
print(groups.aggregate([min,max]))
Output:

      col_1     col_2     col_3    
        min max   min max   min max
col_4                              
100       1   7    10  16    20  26
200       2   8    11  17    21  27
300       3   9    12  18    22  28

```
Concatenation operation:
```python
df3 = pd.DataFrame(['10', '20', '30', '40'], index=['a1', 'b2', 'c3','d4'], columns=['X'])
df4 = pd.DataFrame(['50', '60', '70'], index=['a1', 'b2', 'c3'], columns=['Y'])
print(pd.concat((df3,df4), sort=False))
Output:

      X    Y
a1   10  NaN
b2   20  NaN
c3   30  NaN
d4   40  NaN
a1  NaN   50
b2  NaN   60
c3  NaN   70

```

Concatenate and ignore index:
```python
print(pd.concat((df3,df4), ignore_index= True, sort=False))
Output:
     X    Y
0   10  NaN
1   20  NaN
2   30  NaN
3   40  NaN
4  NaN   50
5  NaN   60
6  NaN   70

```

Join Operation:
```python
print(df3.join(df4))
Output:

     X    Y
a1  10   50
b2  20   60
c3  30   70
d4  40  NaN
```
Merge Operation:
```python
new_ser = pd.Series(['1','2','3'],index=['a1','b2','c3'])
df3['Z'] = new_ser
df4['Z'] = new_ser
print(pd.merge(df3,df4))

Output:
    X  Z   Y
0  10  1  50
1  20  2  60
2  30  3  70

```

```python
print(pd.merge(df3,df4, how='outer'))
Output:

    X    Z    Y
0  10    1   50
1  20    2   60
2  30    3   70
3  40  NaN  NaN

```
Handling missing data, Drop rows with missing values:
```python
print(df3.dropna())
Output:

     X  Z
a1  10  1
b2  20  2
c3  30  3

```
Fill all missing values with some value:
```python
print(df3.fillna(1))
Output:
     X  Z
a1  10  1
b2  20  2
c3  30  3
d4  40  1
```
Read a _.csv_ file:
```python
pd.read_csv('filename.csv')
```

Write to _.csv_ file:
```python
df.to_csv('output.csv')
```

Read excel file:
```python
pd.read_excel('filename.xlsx')
```

Write to excel file:
```python
df.to_excel('filename.xlsx', sheet_name='sheet1')
```

Reading data from a sql table:
```python
from sqlalchemy import create_engine
engine = create_engine('sqlite:///')
pd.read_sql_table('table_name', engine)
pd.read_sql_query("SELECT * FROM table_name;", engine)

```










