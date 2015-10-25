__author__ = 'Vikas'

import pandas as pd
import pylab as pl
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


train = pd.read_csv(r"D:\Kaggle\RossmannSales\RandomForest\train.csv")
store = pd.read_csv(r"D:\Kaggle\RossmannSales\RandomForest\store.csv")
test = pd.read_csv(r"D:\Kaggle\RossmannSales\RandomForest\test.csv")

train = train.drop(train[train.Sales < 1].index)

train = pd.merge(train, store, on='Store', how='outer')

test = pd.merge(test, store, on='Store', how='outer')

train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])

train['Year'] = pd.DatetimeIndex(train['Date']).year
train['Month'] = pd.DatetimeIndex(train['Date']).month
train['Store'] = train['Store'].astype(int)

test['Year'] = pd.DatetimeIndex(test['Date']).year
test['Month'] = pd.DatetimeIndex(test['Date']).month
test['Store'] = test['Store'].astype(int)

train['LogSales'] = np.log10(train['Sales'])

cols = train.columns[1:]

clf = RandomForestClassifier(n_jobs=3, verbose=3)
start = time.time()


train['Assortment'][train.Assortment.str.contains('c')] = 3

features = ['Store','DayOfWeek','Open','Promo','Assortment','CompetitionDistance']

## 'StateHoliday','SchoolHoliday','StoreType','Assortment','CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2','Promo2SinceWeek','Promo2SinceYear','PromoInterval','Year','Month']

clf.fit(train[features], train.LogSales)




