# Environment Setup
## Anaconda & Python (Language)

![anconda_logo](https://upload.wikimedia.org/wikipedia/en/c/cd/Anaconda_Logo.png)

Anaconda Distribution is the world's most popular Python data science platform. This handles all your Python packages, creates environments, and handles all the messy stuff.

### Installation
It is an open-source tool, available [https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/). Ideally, always go for the latest version of Python available. At the moment, this is Python3.7.

## Jupyter (IDE)
![jupyter_logo](https://www.dataquest.io/wp-content/uploads/2019/01/1-LPnY8nOLg4S6_TG0DEXwsg-1.png)

We shall be using Jupyter Notebook as our IDE. This is a web-based interactive computational environment and is one of the most-commonly used tool by Data Scientist. 
### Installation
It is automatically installed via Anaconda.
### Getting Started
It can be started either via terminal ```jupyter notebook``` or via GUI from Anaconda Navigator.
### Alternative IDEs
#### Spyder - https://www.spyder-ide.org/
| | |
|:-------------------------:|:-------------------------:|
|<img width="300" alt="spyder" src="https://www.pngitem.com/pimgs/m/600-6008961_atom-spyder-python-logo-png-transparent-png.png">| <img width="1604" alt="spyder_ide" src="https://steemitimages.com/0x0/https://s3-us-west-2.amazonaws.com/huntimages/production/steemhunt/2018-11-05/23281c4c-spyder.png">|

#### PyCharm - https://www.jetbrains.com/pycharm/
| | |
|:-------------------------:|:-------------------------:|
|<img width="300" alt="pycharm" src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/PyCharm_Logo.svg/1200px-PyCharm_Logo.svg.png">  |  <img width="1604" alt="pycharm_ide" src="https://confluence.jetbrains.com/download/attachments/51188837/pyCharm3.png">|

# Intro to Machine Learning

What does it mean to learn?  Learning is a process where we take a series of observations and draw conclusions based on past experiences. For example, we can learn to recognize patterns in experiential data such as when I take the later bus, I'm late to work.  Machine Learning is when we teach a computer to do the same thing, namely find patterns in data.  The idea is that humans are really great at finding patterns, but relatively slow at looking through large amounts of data.  Computers need to be trained to find the patterns, but they can process data of the sort of we have (csv files, images, etc) incredibly fast.

The revolution of Machine Learning has its roots in two main factors

1. A massive amount of newly generated data
2. A massive improvement in computer memory and performance

If we want to leverage machine learning, we need to learn to teach computers to recognize patterns and leverage that ability to solve real world patterns.  Lets start with a really simple example.

Say we have one dimensional data given by a single feature $X$ and a corresponding set of labels $y$.  We want to model this data, so we will create a relationship \begin{equation} f(X) \approx y .\end{equation} This function $f$ will represent our model.  We will generate the data here by randomly choosing an exponent for a trend and adding some random noise.  Lets create the data and see what this looks like.


```python
import matplotlib
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```


```python
matplotlib.rcParams.update({'font.size': 18,
                            'lines.linewidth' : 3,
                           'figure.figsize' : [15, 5],
                           'lines.markersize': 10})
pd.options.mode.chained_assignment = None
```


```python
X = np.linspace(0, 1, 100)
exp = np.random.choice([2, 3])
y = X**exp + np.random.randn(X.shape[0])/10
plt.plot(X, y, '.');
```


![png](output_5_0.png)


We will now generate the predictive relationship by using one of the simplest possible methods, fitting a line to the data


```python
p = np.polyfit(X, y, 1)
z = np.poly1d(p)
plt.plot(X, y, '.')
plt.plot(X, z(X), label=r"Model: ${:.2f}x + {:.2f}$".format(*p))
plt.plot(X, X**exp, label=r'Truth: $x^{}$'.format(exp))
plt.legend();
```


![png](output_7_0.png)


We now have a model for this data, learned by the computer, namely given an $X$ value (or a bunch of values), we can predict the output.  In the context of Machine Learning, this is called a Linear Regression and is a quite powerful and general method to learn.  Just this example opens up many questions we will be answering in later lectures:

1. How good is the model?
2. Is the model generalizable?
3. What does this model teach us about the data?

Lets start with question 3, which in many ways is the most important question.  For this simple model we can see that the $y$ vector of labels has a positive correlation with the features $X$. 

## Why do we do Machine Learning
Normally the goal of machine learning is two-fold

1. To understand the data we already have
2. Use this understand to make predictions about unlabeled data

Machine Learning falls into two classes, **supervised** learning and **unsupervised** learning.  In supervised learning we are trying to learn a predictive relationship between **features** of our data and some sort of output label. In unsupervised learning we want to find trends in our features without using any target labels. Unsupervised learning typically relies on reducing the dimensionality of the data.  

A human example of supervised learning would be borrowing books from a library on mathematics and geography. By reading different books belonging to each topic, we learn what symbols, images, and words are associated with math, and which are associated with geography. A similar unsupervised task would be to borrow many books without knowing their subject. We can see some books contain similar images (maps) and some books contain similar symbols (e.g. the Greek letters $\Sigma$ and $\pi$). We say the books containing maps are similar and that they are different from the books containing Greek letters. Crucially, _we do not know what the books are about, only that they are similar or different_.

## Some data
Lets introduce a dataset to play with, namely the California housing data. The data set contains the median house value for each census block group in California.



```python
from sklearn.datasets import fetch_california_housing

# get data
data = fetch_california_housing()
X = data['data']
y = data['target']

print(data['DESCR'])
```

    .. _california_housing_dataset:
    
    California Housing dataset
    --------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 20640
    
        :Number of Attributes: 8 numeric, predictive attributes and the target
    
        :Attribute Information:
            - MedInc        median income in block
            - HouseAge      median house age in block
            - AveRooms      average number of rooms
            - AveBedrms     average number of bedrooms
            - Population    block population
            - AveOccup      average house occupancy
            - Latitude      house block latitude
            - Longitude     house block longitude
    
        :Missing Attribute Values: None
    
    This dataset was obtained from the StatLib repository.
    http://lib.stat.cmu.edu/datasets/
    
    The target variable is the median house value for California districts.
    
    This dataset was derived from the 1990 U.S. census, using one row per census
    block group. A block group is the smallest geographical unit for which the U.S.
    Census Bureau publishes sample data (a block group typically has a population
    of 600 to 3,000 people).
    
    It can be downloaded/loaded using the
    :func:`sklearn.datasets.fetch_california_housing` function.
    
    .. topic:: References
    
        - Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
          Statistics and Probability Letters, 33 (1997) 291-297
    


## Supervised and Unsupervised Machine Learning
Lets first talk about supervised learning as that is where we will spend most of our time. Formally, the supervised machine problem can be stated as:
- given a matrix $X$, of dimensions $n \times p$, 
- create a predictive relationship (or function) $f(X)$ where
$$ f(X) \approx y $$ 
    - $y$ is a vector of dimension $n$.  
    - $X$ is referred to as the **feature matrix** and $y$ as the **labels**.


```python
dt_cali = fetch_california_housing()
df_cali = pd.DataFrame(dt_cali.data, columns=dt_cali.feature_names)
df_target = [1 if ii < 2 else 0  for ii in dt_cali['target'] ]
numRows = 10
df_sup = df_cali.head(numRows)
df_sup['-'] = '--------'
df_sup['MedianHouseValue'] = dt_cali['target'][:numRows]
# df_sup['AffordableHouse'] = df_target[:numRows]
print('Supervised Learning Dataset:')
df_sup
```

    Supervised Learning Dataset:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>-</th>
      <th>MedianHouseValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.984127</td>
      <td>1.023810</td>
      <td>322.0</td>
      <td>2.555556</td>
      <td>37.88</td>
      <td>-122.23</td>
      <td>--------</td>
      <td>4.526</td>
    </tr>
    <tr>
      <td>1</td>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.238137</td>
      <td>0.971880</td>
      <td>2401.0</td>
      <td>2.109842</td>
      <td>37.86</td>
      <td>-122.22</td>
      <td>--------</td>
      <td>3.585</td>
    </tr>
    <tr>
      <td>2</td>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.288136</td>
      <td>1.073446</td>
      <td>496.0</td>
      <td>2.802260</td>
      <td>37.85</td>
      <td>-122.24</td>
      <td>--------</td>
      <td>3.521</td>
    </tr>
    <tr>
      <td>3</td>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.817352</td>
      <td>1.073059</td>
      <td>558.0</td>
      <td>2.547945</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>--------</td>
      <td>3.413</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.281853</td>
      <td>1.081081</td>
      <td>565.0</td>
      <td>2.181467</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>--------</td>
      <td>3.422</td>
    </tr>
    <tr>
      <td>5</td>
      <td>4.0368</td>
      <td>52.0</td>
      <td>4.761658</td>
      <td>1.103627</td>
      <td>413.0</td>
      <td>2.139896</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>--------</td>
      <td>2.697</td>
    </tr>
    <tr>
      <td>6</td>
      <td>3.6591</td>
      <td>52.0</td>
      <td>4.931907</td>
      <td>0.951362</td>
      <td>1094.0</td>
      <td>2.128405</td>
      <td>37.84</td>
      <td>-122.25</td>
      <td>--------</td>
      <td>2.992</td>
    </tr>
    <tr>
      <td>7</td>
      <td>3.1200</td>
      <td>52.0</td>
      <td>4.797527</td>
      <td>1.061824</td>
      <td>1157.0</td>
      <td>1.788253</td>
      <td>37.84</td>
      <td>-122.25</td>
      <td>--------</td>
      <td>2.414</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2.0804</td>
      <td>42.0</td>
      <td>4.294118</td>
      <td>1.117647</td>
      <td>1206.0</td>
      <td>2.026891</td>
      <td>37.84</td>
      <td>-122.26</td>
      <td>--------</td>
      <td>2.267</td>
    </tr>
    <tr>
      <td>9</td>
      <td>3.6912</td>
      <td>52.0</td>
      <td>4.970588</td>
      <td>0.990196</td>
      <td>1551.0</td>
      <td>2.172269</td>
      <td>37.84</td>
      <td>-122.25</td>
      <td>--------</td>
      <td>2.611</td>
    </tr>
  </tbody>
</table>
</div>



The general goal of supervised learning is to then apply this model to unlabeled data where can build a feature matrix representative of the original.  This allows us to make predictions! 


```python
df_sup = df_cali.head(numRows+5)[-5:]
df_sup['-'] = '--------'
df_sup['MedianHouseValue'] = '?'
# df_sup['AffordableHouse'] = '?'
df_sup
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>-</th>
      <th>MedianHouseValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>10</td>
      <td>3.2031</td>
      <td>52.0</td>
      <td>5.477612</td>
      <td>1.079602</td>
      <td>910.0</td>
      <td>2.263682</td>
      <td>37.85</td>
      <td>-122.26</td>
      <td>--------</td>
      <td>?</td>
    </tr>
    <tr>
      <td>11</td>
      <td>3.2705</td>
      <td>52.0</td>
      <td>4.772480</td>
      <td>1.024523</td>
      <td>1504.0</td>
      <td>2.049046</td>
      <td>37.85</td>
      <td>-122.26</td>
      <td>--------</td>
      <td>?</td>
    </tr>
    <tr>
      <td>12</td>
      <td>3.0750</td>
      <td>52.0</td>
      <td>5.322650</td>
      <td>1.012821</td>
      <td>1098.0</td>
      <td>2.346154</td>
      <td>37.85</td>
      <td>-122.26</td>
      <td>--------</td>
      <td>?</td>
    </tr>
    <tr>
      <td>13</td>
      <td>2.6736</td>
      <td>52.0</td>
      <td>4.000000</td>
      <td>1.097701</td>
      <td>345.0</td>
      <td>1.982759</td>
      <td>37.84</td>
      <td>-122.26</td>
      <td>--------</td>
      <td>?</td>
    </tr>
    <tr>
      <td>14</td>
      <td>1.9167</td>
      <td>52.0</td>
      <td>4.262903</td>
      <td>1.009677</td>
      <td>1212.0</td>
      <td>1.954839</td>
      <td>37.85</td>
      <td>-122.26</td>
      <td>--------</td>
      <td>?</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('Unsupervised Learning Dataset:')
df_cali.head(numRows)
```

    Unsupervised Learning Dataset:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.984127</td>
      <td>1.023810</td>
      <td>322.0</td>
      <td>2.555556</td>
      <td>37.88</td>
      <td>-122.23</td>
    </tr>
    <tr>
      <td>1</td>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.238137</td>
      <td>0.971880</td>
      <td>2401.0</td>
      <td>2.109842</td>
      <td>37.86</td>
      <td>-122.22</td>
    </tr>
    <tr>
      <td>2</td>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.288136</td>
      <td>1.073446</td>
      <td>496.0</td>
      <td>2.802260</td>
      <td>37.85</td>
      <td>-122.24</td>
    </tr>
    <tr>
      <td>3</td>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.817352</td>
      <td>1.073059</td>
      <td>558.0</td>
      <td>2.547945</td>
      <td>37.85</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.281853</td>
      <td>1.081081</td>
      <td>565.0</td>
      <td>2.181467</td>
      <td>37.85</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <td>5</td>
      <td>4.0368</td>
      <td>52.0</td>
      <td>4.761658</td>
      <td>1.103627</td>
      <td>413.0</td>
      <td>2.139896</td>
      <td>37.85</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <td>6</td>
      <td>3.6591</td>
      <td>52.0</td>
      <td>4.931907</td>
      <td>0.951362</td>
      <td>1094.0</td>
      <td>2.128405</td>
      <td>37.84</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <td>7</td>
      <td>3.1200</td>
      <td>52.0</td>
      <td>4.797527</td>
      <td>1.061824</td>
      <td>1157.0</td>
      <td>1.788253</td>
      <td>37.84</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2.0804</td>
      <td>42.0</td>
      <td>4.294118</td>
      <td>1.117647</td>
      <td>1206.0</td>
      <td>2.026891</td>
      <td>37.84</td>
      <td>-122.26</td>
    </tr>
    <tr>
      <td>9</td>
      <td>3.6912</td>
      <td>52.0</td>
      <td>4.970588</td>
      <td>0.990196</td>
      <td>1551.0</td>
      <td>2.172269</td>
      <td>37.84</td>
      <td>-122.25</td>
    </tr>
  </tbody>
</table>
</div>



I will not be expanding more on Unsupervised Learning. However, I have provided a notebook ```ML99_Clustering``` for any interested parties.


Of course, machine learning is just a tool, one which must be applied with care and thought.  It is not the ideal solution to every problem.  Let us take a look at some of the issues we might find.

## Machine Learning Difficulties

Models can be heavily biased and thus not flexible enough to handle generalization.  Let us plot our original function over a larger range and use the model from before.


```python
X = np.linspace(0, 2, 100)
y = X**exp + np.random.randn(X.shape[0])/10
plt.figure(figsize=(14,6))
plt.plot(X, z(X), label=r"${:.2f}x + {:.2f}$".format(*p))
plt.plot(X, y,'.', label=r'$x^{}$'.format(exp))
plt.legend();
```


![png](output_19_0.png)


The model works fairly well for the range over which initially considered our data, but we can see it will not generalize well to features outside the of the range we considered.  This is a general problem; we should be careful that our training data contains a well sampled distribution over which we expect to make predictions (or we have some prior knowledge that tells us we should be able to extrapolate beyond the domain of our training data).  Machine learning finds patterns in data that it's already seen, and it can't always make good predictions on data it hasn't. 

Lets try to fix this by adding more parameters to the model.


```python
p = np.polyfit(X, y, 15)
z = np.poly1d(p)
plt.figure(figsize=[14, 6])
plt.plot(X, z(X), label=r"${:.2f}x^{{15}} + {:.2f}x^{{14}} + ... + {:.2f}$".format(*p[[0, 1, -1]]))
plt.plot(X, y,'.', label=r'$x^{}$'.format(exp))
plt.legend();
```


![png](output_21_0.png)


Wow looks like a perfect fit!  Maybe too good?  It looks like the model is fitting little wiggles in the data which we know are not real (the actual data is derived from a simple exponent).  Lets try to generalize again.


```python
X = np.linspace(0, 2.5, 100)
y = X**exp + np.random.randn(X.shape[0])/10
plt.figure(figsize=(14,6))
plt.plot(X, z(X), label=r"model")
plt.plot(X, y,'.', label=r'$x^{}$'.format(exp))
plt.legend();
```


![png](output_23_0.png)


Wow again!  That is pretty bad.  This is an example of overfitting, where we have allowed the model too much flexibility and it has fit the noise in the data which is not generalizable.

We will learn more how to combat these issues, but the point is that we need to be careful when choose the model we want to use and the **hyperparameters** (in this case order of the polynomial) for the model.

# Scikit-Learn

In order to perform machine learning we will make use of the `scikit-learn` package will will offer a unified class based interface to different machine learning models and utilities.  

`Scikit-learn` is one of the most popular Python package for machine learning. It has a plethora of machine learning models and provides functions that are often needed for a machine learning workflow. As you will see, it has a nice and intuitive interface. It makes creating complicated machine learning workflows very easy. It is based around the idea of an `Estimator` class which implements the methods necessary for machine learning. Each estimator object will implement a `fit` method which accepts as arguments a feature matrix `X` and a label vector `y` as well as a `predict` method which accepts a an argument a feature matrix `X`.  Lets go through an example.  

First we will need to import the estimator we want, in this case a `LinearRegression` (we only have to do this once per namespace, its just a Python class).


```python
from sklearn.linear_model import LinearRegression
```

Now we can instantiate an instance of this class and pass any hyperparameters into the creation.  The [`LinearRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) has two main hyperparameters, `fit_intercept` and `normalize`.  These have default values, but we will specify them here explicitly.


```python
lr = LinearRegression(fit_intercept=True, normalize=False)
lr
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



Now we can use this object to fit our data from before.  We will use the `fit` method to do this.  We will need to reshape the `X` vector so that its a feature matrix of a single column instead of a one dimensional vector.


```python
lr.fit(X.reshape(-1, 1), y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



The `fit` method will perform the fit and save the fitted parameters internal to the state of the object.  We can see them if we wish.


```python
lr.coef_, lr.intercept_
```




    (array([2.49267196]), -1.0298352363300154)



Saving the parameters inside the instance is extremely useful as it allows us to pickle the entire object and save the parameters inside the model itself.  

Lastly we can use the `predict` method to make predictions. 


```python
predictions = lr.predict(X.reshape(-1, 1))
plt.figure(figsize=(14,6))
plt.plot(X, y, '.', label='data')
plt.plot(X, predictions, label='model')
plt.legend();
```


![png](output_34_0.png)


We will explore linear models in more detail in a later lecture, but if we want to make this model better, we will need to engineer some better features.  To get a sneak peak of where we are going, lets deploy some more `scikit-learn` machinery.


```python
from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(3)
lr = LinearRegression()

X_t = X.reshape(-1,1)
X_t = pf.fit_transform(X_t)
lr_fitted = lr.fit(X_t,y)

predictions = lr_fitted.predict(X_t)

plt.figure(figsize=(14,6))
plt.plot(X, y, '.', label='data')
plt.plot(X, predictions, label='model')
plt.legend();
```


![png](output_36_0.png)


Does this generalize?


```python
X = np.linspace(0, 4, 100)
y = X**exp + np.random.randn(X.shape[0])/10


X_t = X.reshape(-1,1)
X_t = pf.fit_transform(X_t)
lr_fitted = lr.fit(X_t,y)
predictions = lr_fitted.predict(X_t)

plt.figure(figsize=(14,6))
plt.plot(X, y, '.', label='data')
plt.plot(X, predictions, label='model')
plt.legend();
```


![png](output_38_0.png)


## Machine learning models as classes

`Scikit-learn` relies heavily on object-oriented programming principles. It implements machine learning algorithms as classes and users create objects from these "recipes". For example, `Ridge` is a class representing the ridge regression model. To create a `Ridge` object, we simply create an instance of the class. In Python, the convention is that class names use CamelCase, the first letter of each word is capitalized. `Scikit-learn` adopts the convention, making it easy to distinguish what is a class.


```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=0.1)
```

In the above code, we set `alpha=0.1`. Here, `alpha` is a **hyperparameter** of the ridge model. Hyperparameters are model parameters that govern the learning process. In terms of hierarchy, they reside "above" the regular model parameters. They control what values the model parameters are equal to after undergoing training. They can be easily identified as they are the parameters that are set _prior_ to learning. In `scikit-learn`, hyperparameters are set when creating an instance of the class. The default values that `scikit-learn` uses are _usually_ a good set of initial values but this is not always the case. It is important to understand the hyperparameters available and how they affect model performance.

`Scikit-learn` refers to machine learning algorithms as **estimators**. There are three different types of estimators: 
1. classifiers, 
1. regressors, and 
1. transformers. 

Programmatically, `scikit-learn` has a base class called `BaseEstimator` that all estimators inherit. The models inherit an additional class, either `RegressorMixin`, `ClassifierMixin`, and `TransformerMixin`. The inheritance of the second class determines what type of estimator the model represents. We'll divide the estimators into two groups based up on their interface. These two groups are **predictors** and **transformers**.

## Predictors: classifiers and regressors

As the name suggests, predictors are models that make predictions. There are two main methods.

* `fit(X, y)`: trains/fit the object to the feature matrix $X$ and label vector $y$.
* `predict(X)`: makes predictions on the passed data set $X$.


```python
from sklearn.linear_model import LinearRegression

# reload California Dataset
data = fetch_california_housing()
X = data['data']
y = data['target']

# create model and train/fit
model = LinearRegression()
model.fit(X, y)

# predict label values on X
y_pred = model.predict(X)

print(y_pred)
print("shape of the prediction array: {}".format(y_pred.shape))
print("shape of the training set: {}".format(X.shape))
```

    [4.13164983 3.97660644 3.67657094 ... 0.17125141 0.31910524 0.51580363]
    shape of the prediction array: (20640,)
    shape of the training set: (20640, 8)


Note, the output of `predict(X)` is a NumPy array of one dimension. The array has the same size as the number of rows of the data that was passed to the `predict` method. 

Since we are using linear regression and our data has eight features, our model is

$$ y(X) = \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \beta_4 x_4 + \beta_5 x_5 + \beta_6 x_6 + \beta_7 x_7 + \beta_8 x_8 + \beta_0. $$

The coefficients are stored in the fitted model as an object's attribute. `Scikit-learn` adopts a convention where all attributes that are determined/calculated _after_ fitting end in an underscore. The model coefficients and intercept are retrieved using the `coefs_` and the `intercept_` attributes, respectively.


```python
print("β_0: {}".format(model.intercept_))

for i in range(8):
    print("β_{}: {}".format(i+1, model.coef_[i]))
```

    β_0: -36.941920207184324
    β_1: 0.4366932931343243
    β_2: 0.009435778033238106
    β_3: -0.10732204139090407
    β_4: 0.6450656935198111
    β_5: -3.976389421240506e-06
    β_6: -0.003786542654971006
    β_7: -0.42131437752714446
    β_8: -0.43451375467477715


If we wanted to know how well the model performs making predictions with a data set, we can use the `score(X, y)` method. It works by

1. Internally running `predict(X)` to produce predicted values.
1. Using the predicted values to evaluate the model compared to the true label values that were passed to the method.

The evaluation equation varies depending if the model is a regressor or classifier. For regression, it is the $R^2$ value while for classification, it is accuracy.


```python
print("R^2: {:g}".format(model.score(X, y)))
```

    R^2: 0.606233


We used a rather simple model, linear regression. What if we wanted to use a more complicated model? All we need to do is an easy substitution; there is minimum code rewrite as the models have the same interface. Of course, different models have different hyperparameters so we need to be careful when swapping out algorithms. Let's use a more complicated model and train it.


```python
from sklearn.ensemble import GradientBoostingRegressor

# create model and train/fit
model = GradientBoostingRegressor()
model.fit(X, y)

# predict label values on X
y_pred = model.predict(X)

print(y_pred)
print("R^2: {:g}".format(model.score(X, y)))
```

    [4.26432728 3.87864519 3.92074556 ... 0.63664692 0.74759279 0.7994969 ]
    R^2: 0.803324


## Transformers

Transformers are models that process and transform a data set. These transformers are very useful because rarely is our data in a form to feed directly to a machine learning model for both training and predicting. For example, a lot of machine learning models work best when the features have similar scales. All transformers have the same interface:

* `fit(X)`: trains/fits the object to the feature matrix $X$.
* `transform(X)`: applies the transformation on $X$ using any parameters learned
* `fit_transform(X)`: applies both `fit(X)` and then `transform(X)`.

Let's demonstrate transformers with `StandardScaler`, which scales each feature to have zero mean and unit variance. The transformed feature $x'_i$ is equal to

$$ x'_i = \frac{x_i - \mu_i}{\sigma_i}. $$

We'll use pandas to summarize the results of deploying the `StandardScaler` on the California housing data.


```python
from sklearn.preprocessing import StandardScaler

# create and fit scaler
scaler = StandardScaler()
scaler.fit(X)

# scale data set
Xt = scaler.transform(X)

# create data frame with results
stats = np.vstack((X.mean(axis=0), X.var(axis=0), Xt.mean(axis=0), Xt.var(axis=0))).T
feature_names = data['feature_names']
columns = ['unscaled mean', 'unscaled variance', 'scaled mean', 'scaled variance']

df = pd.DataFrame(stats, index=feature_names, columns=columns)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unscaled mean</th>
      <th>unscaled variance</th>
      <th>scaled mean</th>
      <th>scaled variance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MedInc</td>
      <td>3.870671</td>
      <td>3.609148e+00</td>
      <td>6.609700e-17</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>HouseAge</td>
      <td>28.639486</td>
      <td>1.583886e+02</td>
      <td>5.508083e-18</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>AveRooms</td>
      <td>5.429000</td>
      <td>6.121236e+00</td>
      <td>6.609700e-17</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>AveBedrms</td>
      <td>1.096675</td>
      <td>2.245806e-01</td>
      <td>-1.060306e-16</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Population</td>
      <td>1425.476744</td>
      <td>1.282408e+06</td>
      <td>-1.101617e-17</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>AveOccup</td>
      <td>3.070655</td>
      <td>1.078648e+02</td>
      <td>3.442552e-18</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Latitude</td>
      <td>35.631861</td>
      <td>4.562072e+00</td>
      <td>-1.079584e-15</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Longitude</td>
      <td>-119.569704</td>
      <td>4.013945e+00</td>
      <td>-8.526513e-15</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


