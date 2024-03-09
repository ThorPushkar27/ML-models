#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the numpy libraries.
import numpy as np


# # Linear Regression

# In[37]:


# Objects instance value is stored in the self keyword.

class Linear_Regression():
    
    #Initiating the parameters
    
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
    
    
    def fit(self,X,Y):
        
      # Number of the training examples and no. of features.
        self.m, self.n = X.shape # No. of rows and columns.
        
        # Initiating the weight and bias.
        self.w = np.zeros(self.n)   # [0,0,0,0,0,0,0,0]
        self.b = 0
        self.X = X
        self.Y = Y
        
        #Implementing the Gradient Descent in Linear Regression.
        for i in range(self.iterations):
            self.update_weights()
        
    def update_weights(self):
        Y_pred = self.predict(self.X)
        
        # Calculate the gradients
        
        # In dw dot product implicitly handles np.sum function.
        dw = -(2*(self.X.T).dot(self.Y - Y_pred))/self.m
        db = -(2*np.sum(self.Y - Y_pred))/ self.m 
        
        # Updating the weights and bias
        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db
        
        
    def predict(self,X):
        return X.dot(self.w)+ self.b


# Using Linear Regression model for Prediction.

# In[38]:


# Importing the libraries.
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[39]:


salary = pd.read_csv("salary_data.csv")
salary.head()


# In[40]:


salary.shape # no. of rows and columns in the dataframe.


# In[41]:


# Checking for the missing values.
salary.isnull().sum()


# Splitting the data into features and target columns.
# 

# In[42]:


X = salary.drop(columns = "Salary")
Y = salary["Salary"]
X


# In[43]:


Y


# Splitting the data into training and testing data.

# In[45]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.33,random_state= 2) 


# Training the Linear Regression Model.

# In[63]:


model = Linear_Regression(learning_rate = 0.01, iterations = 100)


# In[64]:


model.fit(X_train, Y_train)


# In[65]:


#printing the parameter values.
print("Weight value is:", model.w[0])
print("Bias value is:", model.b)


# Predicting the salary values of the test data.

# In[66]:


test_data_prediction = model.predict(X_test)


# In[67]:


print(test_data_prediction)


# Visualizing the predicted values and the actual values.

# In[68]:


plt.scatter(X_test, Y_test, color="red")
plt.plot(X_test, test_data_prediction, color="blue")
#plt.xlabel('Work Experience')
#plt.ylabel('Salary')
plt.title("Salary vs Work Experience")
plt.show()

