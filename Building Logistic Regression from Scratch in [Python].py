#!/usr/bin/env python
# coding: utf-8

# In[28]:


# Importing the dependencies.
import numpy as np


# # Logistic Regression

# In[48]:


class Logistic_Regression():
    
    # Declearing Learning rate and iterations(Hyperparameters).
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
        
    #fit function to train the model with the dataset.
    def fit(self, X,Y):
        
        self.m, self.n = X.shape # m is no. of rows and n is no. of columns.
        
        # Initiating weight value and the bias value.
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        
        # Implementing Gradient Descent in Logistic Regression.
        for i in range(self.iterations):
            self.update_weight()
        
    def update_weight(self):
        
        # Y_hat formula(Sigmoid Formula)
        Y_hat = 1/(1 + np.exp(-(self.X.dot(self.w)+ self.b)))
         
        # Writting the dw and db formulas.
        dw = (1/self.m) * np.dot(self.X.T,(Y_hat - self.Y))
        db = (1/self.m) * np.sum(Y_hat - self.Y)
        
        
        #Gradient Descent Equations Updations.
        self.w = self.w - self.learning_rate*dw
        
        self.b = self.b - self.learning_rate*db
        
    # Sigmoid Equation and Decision Boundary(either 0 or 1).  
    def predict(self,X):
        Y_pred =  1/(1 + np.exp(-(X.dot(self.w)+ self.b)))
        Y_pred = np.where(Y_pred > 0.5, 1 , 0)
        return Y_pred
         
    


# # Testing Our Logistic Regression code

# In[49]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Data Collection and Analysis

# In[50]:


diabetes = pd.read_csv("diabetes.csv")
diabetes.head()


# In[51]:


diabetes.shape


# In[52]:


diabetes.describe()


# In[53]:


diabetes["Outcome"].value_counts()


#  0 --> Non-diabetic,
#  1 --> diabetic

# In[54]:


diabetes.groupby("Outcome").mean()


# In[55]:


features = diabetes.drop(columns= "Outcome", axis=1)
target = diabetes["Outcome"]


# In[56]:


features


# In[57]:


target


# In[60]:


# Data Standardization.
scaler = StandardScaler()
scaler.fit(features)
standardized_data = scaler.transform(features)


# In[61]:


print(standardized_data)


# In[62]:


X = standardized_data
Y = diabetes["Outcome"]


# In[63]:


X


# Train- Test Split

# In[64]:


X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 2)


# In[65]:


print(X.shape, X_train.shape, X_test.shape)


# # Training The Model

# In[66]:


classifier = Logistic_Regression(learning_rate = 0.01, iterations = 1000)


# In[67]:


# training the logistic Regression model
classifier.fit(X_train, Y_train)


# Model Evaluation

# In[68]:


# Accuracy Score


# In[69]:


X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[70]:


print("The Accuracy score of the model is:", training_data_accuracy)


# In[71]:


X_test_prediction = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[72]:


print("The Accuracy score of the testing model is:", testing_data_accuracy)

