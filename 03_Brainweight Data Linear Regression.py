#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

plt.rcParams['figure.figsize'] = (20.0, 10.0)

data = pd.read_csv(r'C:\Users\Swami\Desktop\Datasets\headbrain.csv')

X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values
m = len(X)
X = X.reshape((m,1))
reg = LinearRegression()
reg = reg.fit(X, Y)
Y_pred = reg.predict(X)

r2_score = reg.score(X, Y)
print(r2_score)

mean_x = np.mean(X)
mean_y = np.mean(Y)
numer = 0
denom = 0
for i in range(m):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
b1 = numer / denom
b0 = mean_y - (b1*mean_x)
print(b1,b0)

max_x = np.max(X) + 100
min_x = np.min(X) - 100

x = np.linspace(min_x, max_x, 1000)
y = b0 + x * b1

plt.scatter(X,Y)
plt.plot(x,y)
plt.legend()
plt.show()

