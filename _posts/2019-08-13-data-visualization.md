---
title: 'Data Visualization with Matplotlib, Plotly and Seaborn'
date: 2019-08-13
permalink: /posts/2019/08/data-visualization/
tags:
  - Artificial Intelligence
  - Python
---

Import all the required libraries:

```python
import cufflinks as cf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.offline as plyo
import seaborn as sns

from mpl_toolkits import mplot3d
from sklearn.datasets import load_digits

mpl.rcParams['font.family'] = 'sans-serif'
%matplotlib inline

plyo.init_notebook_mode(connected=True)
plt.style.use('seaborn')

sns.set(font_scale=1.1)
sns.set_style("whitegrid")

```

Now let's create some interactive plots with Plotly.

```python
#Create a new DataFrame 
data = (np.random.standard_normal((50,5)).cumsum(axis=0))*2 + 200
date = pd.date_range('2019-1-2', freq='B', periods=len(data))
df = pd.DataFrame(data, columns=['AAPL', 'GOOG', 'XYZ', 'ABCD', 'DEFG'], index=date)
plyo.plot(df.iplot(asFigure=True), filename="plots/plotly.html",auto_open=False)
```
<iframe width="100%" height="500" src="/images/plots/plotly.html">stocks</iframe>
<br />
Histogram:
```python
plyo.plot(df.iplot(kind='hist', subplots=True,  bins=15,  asFigure=True), filename="plots/hist_plotly.html",auto_open=False)
```
<iframe width="100%" height="500" src="/images/plots/hist_plotly.html">stocks histogram</iframe>
<br />

Candlestick chart:
```python
df_st = pd.read_csv('data.csv', index_col=0, parse_dates=True)
quotes = df_st[['OpenAsk', 'HighAsk', 'LowAsk', 'CloseAsk']]
quotes = quotes.iloc[-100:]
qf = cf.QuantFig(quotes, title='EUR/USD', legend='top', name='Exchange rate')
plyo.plot(qf.iplot(asFigure=True), filename="plots/exchange_rate.html",auto_open=False)
```
<iframe width="100%" height="500" src="/images/plots/exchange_rate.html">Exchange Rate</iframe>
<br />

Let's create some plots with Matplotlib. A simple plot with one 'y-axis':

```python
#Generate a 2D array with 4 rows and 2 columns
y = np.random.standard_normal((4,2))

#Print the generated numbers.
print(y)

#Manage the size
plt.figure(figsize=(6,4))

#Turn off the grid
plt.grid(False)

#Blue line for first column
plt.plot(y[:,0], lw=1, label='1st col')

#blue circle
plt.plot(y[:,0], 'bo')


#Red line for second column
plt.plot(y[:,1], lw=1, label='2nd col')

#blue circle
plt.plot(y[:,1], 'ro')

#x-axis label
plt.xlabel('x-axis label')

#y-axis label
plt.ylabel('y-axis label')

#Legend with location
plt.legend(loc=0)

#Title
plt.title('Plot')

#Save the plot.
plt.savefig('plots/singleY.png', bbox_inches='tight')
```
```python
Output:

[[ 1.21176579 -0.06254173]
 [-2.23790818 -0.98232992]
 [-0.90849221 -0.25523951]
 [-0.67613312  0.05573889]]
```
![png](/images/plots/singleY.png)
<br />

Plot with Two 'y-axis':
```python
#Create figure and axis objects.
fig, ax1 = plt.subplots()

#Blue line for first column
plt.plot(y[:,0], 'b', lw=1, label='1st col')
plt.plot(y[:,0], 'bo')

plt.xlabel('x-axis label')
plt.ylabel('First y-axis')
plt.legend(loc=3)
plt.grid(False)

#Create a second y-axis that shares the x-axis.
ax1.twinx()

#Red line for second column
plt.plot(y[:,1], 'r', lw=1, label='2nd col')
plt.plot(y[:,1], 'ro')
plt.ylabel('Second y-axis')
plt.legend(loc=1)
plt.grid(False)

plt.title('Plot Name')

#Save the plot.
plt.savefig('plots/two_y_axis.png', bbox_inches='tight')
```
![png](/images/plots/two_y_axis.png)


<br />
Stacked Histogram:
```python
y_hist = np.random.standard_normal((100,2))
plt.figure(figsize=(6,4))
plt.hist(y_hist, label=['1st', '2nd'], color= ['b', 'y'], stacked=True, bins=20, alpha=0.5)
plt.xlabel('value')
plt.ylabel('frequency')
plt.legend(loc=0)
plt.grid(False)

#Save the plot.
plt.savefig('plots/stacked_histogram.png', bbox_inches='tight')
```
![png](/images/plots/stacked_histogram.png)

<br />

Box Plot
```python
fig, ax = plt.subplots(figsize=(6,4))
plt.boxplot(y)
#set the property, xticklabels, after plot has been created
plt.setp(ax, xticklabels=['1st', '2nd'])
#Save the plot.
plt.savefig('plots/box.png', bbox_inches='tight')
```
![png](/images/plots/box.png )

<br />
Bar plot:
```python
y_bar = np.random.rand(20)
plt.bar(np.arange(len(y_bar)), y_bar, width=0.9, color='y', label='2nd')

#Save the plot.
plt.savefig('plots/bar.png', bbox_inches='tight')
```
![png](/images/plots/bar.png )

<br />
3D plot:
```python
cm = plt.get_cmap("RdYlGn")

fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection='3d')
x = np.random.rand(50)
y = np.random.rand(50)
z = np.random.rand(50)
col = np.arange(50)

ax.scatter3D(x,y,z,s=20, c=col, marker='o',cmap=cm)

#Save the plot.
plt.savefig('plots/3d.png', bbox_inches='tight')

```
![png](/images/plots/3d.png )

<br />
```python
ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
#Save the plot.
plt.savefig('plots/virdis.png', bbox_inches='tight')
```
![png](/images/plots/virdis.png )

<br />
Plot MNIST data:
```python
digits = load_digits()
fig = plt.figure(figsize= (5,5))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(64):
    plt.subplot(8,8,i+1)
    plt.imshow(digits.images[i], cmap = plt.cm.binary, interpolation='nearest')
    plt.text(0,0, digits.target[i], horizontalalignment='left',verticalalignment='top')
    plt.xticks([]),plt.yticks([])
```

<img src="/images/plots/mnist.png" align="middle"  alt="MNIST" height="1500" width="50%">
<br />

Let's try Seaborn library now. Scatter plot:
```python
df_iris = sns.load_dataset('iris')
sns.regplot(x=df_iris["sepal_length"], y=df_iris["sepal_width"], fit_reg=False)

#Save the plot.
plt.savefig('plots/scatter.png', bbox_inches='tight')
```
![png](/images/plots/scatter.png )

<br />
Heat Map:
```python
uniform_data = np.random.rand(10, 12)
hm = sns.heatmap(uniform_data)
heat_map=hm.get_figure()
heat_map.savefig("plots/heat_map.png")
```
![png](/images/plots/heat_map.png )

<br />
Heat Map of _flights_ dataset. 
```python
flights = sns.load_dataset("flights")
flights = flights.pivot("month", "year", "passengers")
hm_flights = sns.heatmap(flights)
heat_map_flights = hm_flights.get_figure()
heat_map_flights.savefig("plots/heat_map_flights.png")

```
![png](/images/plots/heat_map_flights.png )


















