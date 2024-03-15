#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the dependencies.
import numpy as np


# # Lasso Regression.

# In[1]:


class Lasso_Regression():
    
    #Initiating the hyperparameters.
    def __init__(self, learning_rate, iterations, lambda_parameter):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_parameter = lambda_parameter
        
    # Fit function
    def fit(self, X,Y):
        # m,n is the no. of rows and columns in X 
        self.m, self.n = X.shape
        
        # Initiating the parameter w and b
        self.w = np.zeros(self.n)
        self. b = 0
        self.X = X
        self.Y = Y
        
        # Implementing the Gradient Descent algorithm for Optimization.
        for i in range(iterations):
            update_weigth()
        
    
    #function for updating weight and bias values
    def update_weight(self):
        # Linear Equation of the model
        Y_prediction = self.predict(self.X)
        
        # Gradients(dw)
        dw = np.zeros(self.n)
        
        for i in range (self.n):
            if(self.w[i]>0):
                
                dw[i] = (-(2*(self.X[:,i]).dot(self.Y - Y_prediction)) + self.lambda_parameter)/self.m
                
            else:
                
                dw[i] = (-(2*(self.X[:,i]).dot(self.Y - Y_prediction))-self.lambda_parameter)/self.m
                
         
        # Gradient for bias.
        db = -2 * np.sum(self.Y - Y_prediction) / self.m
        
        #Updating the weigths and bias values.
        
        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db
        
                
        
    
    
    # Predicting the target variable
    def predict(self,X):
        
        return X.dot(self.w)+self.b
        
    


# In[ ]:




