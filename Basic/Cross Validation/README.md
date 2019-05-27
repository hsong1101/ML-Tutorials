
# Cross Validation 

## Intro

Every time after we make a model, we have to check if it is functioning properly or not. But testing on the data that was used to build the model is a bad idea. When testing if one performs well, it should be tested on a separate data set which hasn't been used for training which is the reason before doing anything, we separate the whole data into training and testing sets.

Once we build a model, we test it on testing set over and over again until it becomes accurate enough to be used. <br>
A problem in this case is this is also a bad practice. Doing this will cause the model to be overfitting on testing data that when some new data is inserted, it may work poorly. This is a reason that testing set should only be used once throughout the entire training and testing. 

If we cannot use the testing set, how can we check if a model is working well? We cannot use training nor testing set. For this reason we have something called Cross Validation. This is a simple idea. We just divide a training set into $k$ sets and train a model with $k-1$ set and get an error value with the remaining set. After we divided and used the last set to train, next thing we do is initialize a new model and train it with different combinations of $k-1$ sets and test with the one left. We do this process until there is no more combination of sets we haven't used to train model. Now we have $k$ different error values. We get the mean value of it and that will become our final performace score (or error value) of the model. Also by doing cross validation, we can check which subset of features produce high or low error and select the ones with low value.<br>
The following is a pseudo-code of cross validation.
function(features, x, y, k)

error = []

  for k times
    size_of_each_set = ceiling(length(x) / k)
    
    validation = x[length(x) - size :]
    v_label = y[length(x) - size : ]
    
    training = x[:length(x) - size]
    t_label = y[:length(x) - size]
    
    model = init
    model.train(training, t_label)
    
    pred = model.predict(validation)
    error = append(loss(v_label), pred)

return mean of error
In above code, the x is the whole training set (not the whole data) and y is the label corresponding to it.

## Coding

We can easily do the cross validation with existing library (scikit). But since the idea is simple and implementation is also easy do, let's try implementing it on our own first, and then see how to use the library. <br>
Again, we will use linear regression with housing price data.

### Import libraries and Load and Prep Data


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```


```python
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
```


```python
X = train
y = train.loc[:, ['SalePrice']]
```


```python
# set split size to 0.8
X_train, y_train, X_test, y_test = train_test_split(X, y, train_size=0.8)
```

    c:\users\hsong1101\anaconda3\envs\tensorflow\lib\site-packages\sklearn\model_selection\_split.py:2026: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
      FutureWarning)
    

Now, let's define necessary functions.


```python
# Root Mean Squared Error
def rmse(y, y_hat):
    return np.sqrt(np.mean((y - y_hat)**2))

def get_index(x, size):
    index = []
    validation = []
    r = set([i for i in range(len(x))])
    
    for i in range(len(x)):
        validation.append(i)

        if (i+1) % size == 0 or i == len(x)-1:
            
            training = list(r - set(validation))
            index.append([np.array(training), np.array(validation)])
            validation = []

    return index

def cross_validation(x, y, k=5):
    
    size = ceil(len(x) / k)
    
    # list of list of indices to split
    index = get_index(x, size)
    error = np.array([])
    
    model = LinearRegression()
    
    for training, validation in index:
        
        train = x.iloc[training]
        test = y.iloc[training]
        
        val_train = x.iloc[validation]
        val_test = y.iloc[validation]
        
        model.fit(train, test)

        pred = model.predict(val_train)
        
        err = rmse(val_test, pred)
        
        error = np.append(err, error)
    
    return error.mean()

```

Now that we have finished all the functions necessary for cross validation, let's check out a set of features to put into the model to compare each performance. <br>
I've set five different list of features as below and using the cross validation functions we just implemented, we will see which features set produce the least amount of error.


```python
features = [['LotArea', 'YearBuilt', 'FullBath'],
['LotArea', 'GarageArea', 'PoolArea'],
['GarageArea', 'FullBath', 'HalfBath'],
['LotArea', 'YearBuilt', 'GarageArea'],
['LotArea', 'GarageArea', 'FullBath']]
```


```python
errors = []
best_feature_set = None
least_error = 0

for f in features:
    errors.append(cross_validation(X[f], y))
    
least_error = min(errors)
best_feature_index = errors.index(least_error)
best_feature_set = features[best_feature_index]

best_feature_set
```




    ['GarageArea', 'FullBath', 'HalfBath']




```python
for f, loss in zip(features, errors):
    print('\nFeatures: {} with Loss: {}'.format(f, loss))
    
print('\n\nBest Features: {}'.format(best_feature_set))
```

    
    Features: ['LotArea', 'YearBuilt', 'FullBath'] with Loss: 60177.62086438129
    
    Features: ['LotArea', 'GarageArea', 'PoolArea'] with Loss: 61322.452532656185
    
    Features: ['GarageArea', 'FullBath', 'HalfBath'] with Loss: 54635.92474394878
    
    Features: ['LotArea', 'YearBuilt', 'GarageArea'] with Loss: 57546.42623587614
    
    Features: ['LotArea', 'GarageArea', 'FullBath'] with Loss: 55443.39583203288
    
    
    Best Features: ['GarageArea', 'FullBath', 'HalfBath']
    

It turns out, set of GarageAre and number of full and half baths result in the least error. With this information, we have extracted features to train and evaluate a model.

The main reason we use the validation sets is to avoid using test sets multiple times which could result in overfitting data. If we don't use it, we will keep trying to improve the model by fitting it to the testing sets, rather than general data. In every case, the testing sets should only be used once at the end of the whole process. 

Since the cross validation is simple and easy to implement, I've done so myself but there already is existing library for cross validation (from scikit) such as the KFold in the following.


```python
from sklearn.model_selection import KFold
```

What it does is to replace the get_index function we created. It can be substituted just like the next.


```python
def cross_validation(x, y, k=5):
    
    size = ceil(len(x) / k)
    
    # n_splits is the number to split the data
    kf = KFold(n_splits=5)
    
    error = np.array([])
    
    model = LinearRegression()
    
    # we should put data in kf.split and it would return the training and validation indices just as we created.
    for training, validation in kf.split(x):
        
        train = x.iloc[training]
        test = y.iloc[training]
        
        val_train = x.iloc[validation]
        val_test = y.iloc[validation]
        
        model.fit(train, test)

        pred = model.predict(val_train)
        
        err = rmse(val_test, pred)
        
        error = np.append(err, error)
    
    return error.mean()
```

## Ending 

As I've mentioned above already, it is very important to use the testing set only at the last step to check overall performance of a model or else it could fall into the overfitting.

We could use cross validation to see which combinations of features could produce better performance than others and which ones are worse. Doing so, we can eliminate any feature whose absence doesn't impact much on the model to reduce the dimensionality of data. Also we can try out different hyperparameter values with the best feature sets to further improve the model. 

Thanks again for reading!
