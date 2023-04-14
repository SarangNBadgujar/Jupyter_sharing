#!/usr/bin/env python
# coding: utf-8

# In[91]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 #to run library in jupyter
get_ipython().run_line_magic('matplotlib', 'inline')
import math

titanic_data = pd.read_csv(r'C:\Users\Swami\Desktop\Datasets\Titanic.csv')
titanic_data.head(10)


# In[4]:


titanic_data.shape


# In[6]:


titanic_data.size


# In[7]:


print("# of passengers in original data : " +str(len(titanic_data)))


# In[8]:


#Analyzing data

sns.countplot(x = "Survived" , data = titanic_data)


# In[10]:


sns.countplot(x = "Survived" , hue = "Sex" , data = titanic_data)


# In[13]:


sns.countplot(x = "Survived" , hue = "Pclass" , data = titanic_data)


# In[14]:


titanic_data["Age"].plot.hist()


# In[16]:


titanic_data["Fare"].plot.hist(bins = 20, figsize = (10,5))


# In[18]:


titanic_data.info()


# In[19]:


sns.countplot(x = "SibSp" , data = titanic_data)


# In[20]:


# Data Wrangling

titanic_data.isnull()


# In[21]:


titanic_data.isnull().sum()


# In[28]:


sns.heatmap(titanic_data.isnull(), yticklabels=False, cmap = 'viridis')


# In[29]:


sns.boxplot(x="Pclass", y = "Age", data = titanic_data)


# In[30]:


titanic_data.head(5)


# In[31]:


titanic_data.drop("Cabin",  axis=1, inplace=True)


# In[32]:


titanic_data.head(5)


# In[33]:


titanic_data.dropna(inplace=True)


# In[34]:


sns.heatmap(titanic_data.isnull(), yticklabels=False, cmap = 'viridis')


# In[35]:


titanic_data.isnull().sum()


# In[36]:


titanic_data.head(2)


# In[41]:


sex=pd.get_dummies(titanic_data['Sex'], drop_first=True)
sex.head(5)


# In[43]:


embark = pd.get_dummies(titanic_data["Embarked"], drop_first=True)
embark.head(5)


# In[44]:


Pclass = pd.get_dummies(titanic_data["Pclass"], drop_first=True)
Pclass.head(5)


# In[50]:


titanic_data=pd.concat([titanic_data,sex,embark,Pclass],axis=1)
titanic_data.head(5)


# In[60]:


titanic_data.drop(["Sex","Embarked","Name","PassengerId","Pclass","Ticket"], axis=1, inplace=True)


# In[61]:


titanic_data.head(5)


# In[70]:


# Training and Testing Data

X = titanic_data.drop("Survived", axis=1)
y = titanic_data["Survived"]


# In[96]:


from sklearn.model_selection import train_test_split
#earlier sklearn.cross_validation
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression(solver='lbfgs', max_iter=1000)


# In[97]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

logmodel.fit(X_train,y_train)
# In[98]:


logmodel.fit(X_train,y_train)


# In[99]:


predictions = logmodel.predict(X_test)


# In[102]:


from sklearn.metrics import classification_report


# In[103]:


classification_report(y_test,predictions)


# In[104]:


from sklearn.metrics import confusion_matrix


# In[107]:


confusion_matrix(y_test,predictions)


# In[108]:


from sklearn.metrics import accuracy_score


# In[109]:


accuracy_score(y_test,predictions)

