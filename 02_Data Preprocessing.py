#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

dataset = pd.read_csv(r'C:\Users\Swami\Desktop\Datasets\Data.csv')

dataset.shape

x = dataset[['Country','Age','Salary']].values
y = dataset[['Purchased']].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

imputer = imputer.fit(x[:,1:3])

x[:,1:3] = imputer.transform(x[:,1:3])

from sklearn.preprocessing import LabelEncoder

label_encode_x = LabelEncoder()

x[:,0]=label_encode_x.fit_transform(x[:,0])

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
onehotencoder.fit_transform(dataset.Country.values.reshape(-1,1)).toarray()

labelencoder_y = LabelEncoder()

y=labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler

sc_x=StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)

x_train

