#!/usr/bin/env python
# coding: utf-8

# # HyperParameter Tuning

# In[2]:


# Importing the dependencies
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# We will be working on the breast cancer dataset

# In[3]:


breast = sklearn.datasets.load_breast_cancer()


# In[4]:


print(breast)


# In[5]:


data_frame = pd.DataFrame(breast.data, columns= breast.feature_names)


# In[7]:


data_frame.head()


# In[8]:


# Adding the target column to the dataframe
data_frame["label"] = breast.target


# In[9]:


data_frame.head()


# In[10]:


data_frame.shape


# In[11]:


data_frame.isnull().sum()


# In[12]:


data_frame["label"].value_counts()


# 1 --> Benign
# 2 --> Malignant

# Seperating the features and the target variables

# In[13]:


X = data_frame.drop(columns="label", axis =1)
Y = data_frame["label"]


# In[15]:


X.shape


# In[16]:


Y.shape


# In[17]:


X = np.asarray(X)
Y = np.asarray(Y)


# # GridSearchCV

# Used for determining the best parameter for our model.

# In[27]:


# Loading the SVC Classifier.
model = SVC()


# In[28]:


# HYPERPARAMETERS


# In[29]:


parameters = {
    "kernel": ["linear","poly","rbf","sigmoid"],
    'C': [1,5,10,20]
}


# In[30]:


# GridSearchCV 
classifier = GridSearchCV(model,parameters, cv=5)


# In[32]:


# fitting the data to our model.
classifier.fit(X,Y)


# In[33]:


classifier.cv_results_


# In[34]:


# Best Parameter
best_parameter = classifier.best_params_


# In[35]:


best_parameter


# In[36]:


# Highest Accuracy
highest_accuracy = classifier.best_score_


# In[37]:


highest_accuracy


# In[38]:


# loading the results to the pandas dataframe
results = pd.DataFrame(classifier.cv_results_)


# In[39]:


results.head()


# In[40]:


grid_search_result = results[["param_C","param_kernel","mean_test_score"]]
grid_search_result.head()


# # RandomizedSearchCV

# In[41]:


# Loading the SVC Classifier.
model = SVC()


# In[42]:


parameters = {
    "kernel": ["linear","poly","rbf","sigmoid"],
    'C': [1,5,10,20]
}


# In[43]:


classifier =RandomizedSearchCV(model,parameters, cv=5)


# In[44]:


# fitting the data to our model.
classifier.fit(X,Y)


# In[45]:


classifier.cv_results_


# In[46]:


# Best Parameter
best_parameter = classifier.best_params_


# In[47]:


best_parameter


# In[48]:


# Highest Accuracy
highest_accuracy = classifier.best_score_


# In[49]:


highest_accuracy


# In[50]:


# loading the results to the pandas dataframe
results = pd.DataFrame(classifier.cv_results_)


# In[51]:


results.head()


# In[54]:


randomized_search_result = results[["param_C","param_kernel","mean_test_score"]]
randomized_search_result


# In[ ]:




