#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import pandas as pd


# In[41]:


dataset = pd.read_csv(r'C:\Users\Swami\Desktop\Datasets\Data.csv')


# In[42]:


dataset


# In[43]:


dataset.shape


# In[44]:


x = dataset[['Country','Age','Salary']].values


# In[45]:


x


# In[46]:


y = dataset[['Purchased']].values


# In[47]:


y


# In[48]:


from sklearn.impute import SimpleImputer


# In[49]:


imputer = SimpleImputer(missing_values=np.nan,strategy='mean')


# In[50]:


imputer = imputer.fit(x[:,1:3])


# In[51]:


x[:,1:3] = imputer.transform(x[:,1:3])


# In[52]:


x


# In[53]:


from sklearn.preprocessing import LabelEncoder


# In[54]:


label_encode_x = LabelEncoder()


# In[55]:


x[:,0]=label_encode_x.fit_transform(x[:,0])


# In[56]:


x


# In[57]:


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
onehotencoder.fit_transform(dataset.Country.values.reshape(-1,1)).toarray()


# In[58]:


labelencoder_y = LabelEncoder()


# In[59]:


y=labelencoder_y.fit_transform(y)


# In[60]:


y


# In[61]:


from sklearn.model_selection import train_test_split


# In[62]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[63]:


x_train


# In[64]:


x_test


# In[65]:


y_train


# In[66]:


y_test


# In[67]:


from sklearn.preprocessing import StandardScaler


# In[68]:


sc_x=StandardScaler()


# In[81]:


x_train = sc_x.fit_transform(x_train)


# In[82]:


x_test = sc_x.transform(x_test)


# In[83]:


x_train


# In[ ]:




