#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the dependencies.
import numpy as np


# # Support Vector Machine(SVM) Classifier.

# In[2]:


class SVM_classifier():
    
    # Initiating the hyperparameters.
    def __init__(self, learning_rate, iterations, lambda_parameter):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_parameter = lambda_parameter
        
    # fitting the dataset to the SVM classifier.
    def fit(self, X,Y):
        # m --> no. of features, n--> no. of datapoints.
        self.m, self.n  = X.shape
        
        # Initiating the value of w and b
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y 
        
        # Implementing the Gradient Descent Algorithm for Optimization.
        for i in range(self.iterations):
            self.update_weights()
            
        
    
    # Updating the value of w and b in the Gradient Descent function.
    def update_weights(self):
        # Label Encoding.
        y_label = np.where(self.Y <=0 , -1, 1)
        
        for index,x_i in enumerate(self.X):
            condition = y_label[index]* (np.dot(x_i,self.w) - self.b) >=1
            
            if(condition == True):
                dw = 2 * self.lambda_parameter * self.w
                db = 0
            else:
                dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])
                db = y_label[index]
                
            self.w = self.w - self.learning_rate*dw
            self.b = self.b - self.learning_rate*db
          

    #Predicting the Output.
    def predict(self,X):
        output = np.dot(X,self.w) - self.b
        predicted_labels = np.sign(output)
        
        # Reverting back the labels -1 to 0 and 1 remains as 1.
        y_hat = np.where(predicted_labels<= -1, 0, 1)
        return y_hat
    


# # Testing the SVM classifier

# In[4]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Data Collection and Analysis

# In[5]:


diabetes = pd.read_csv("diabetes.csv")
diabetes.head()


# In[6]:


diabetes.shape


# In[7]:


diabetes.describe()


# In[8]:


diabetes["Outcome"].value_counts()


#  0 --> Non-diabetic,
#  1 --> diabetic

# In[9]:


diabetes.groupby("Outcome").mean()


# In[10]:


features = diabetes.drop(columns= "Outcome", axis=1)
target = diabetes["Outcome"]


# In[11]:


# Data Standardization.
scaler = StandardScaler()
scaler.fit(features)
standardized_data = scaler.transform(features)


# In[12]:


print(standardized_data)


# In[13]:


X = standardized_data
Y = diabetes["Outcome"]


# Train_Test_Split

# In[14]:


X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 2)


# In[15]:


print(X.shape, X_train.shape, X_test.shape)


# # Training The Model

# In[16]:


classifier = SVM_classifier(learning_rate = 0.001, iterations = 1000, lambda_parameter = 0.01)


# Training the SVM classifier with the training data.

# In[17]:


classifier.fit(X_train, Y_train)


# # Model Evaluation.
# 

# In[18]:


X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[20]:


print("The Accuracy score of the model is:", training_data_accuracy)


# In[21]:


X_test_prediction = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[22]:


print("The Accuracy score of the testing model is:", testing_data_accuracy)


# # Building the Predictive System.

# In[27]:


input_data = [0,137,40,35,168,43.1,2.288,33]

#change the input data to the numpy array
input_data_as_numpy_array = np.asarray(input_data)
#reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


# Standardizing the input_data
standard_data =scaler.transform(input_data_reshaped)

#print(standard_data)

prediction = classifier.predict(standard_data)
#print(prediction)

if(prediction[0] == 1):
    print("The Person is Diabetic.")
else:
    print("The Person is Non-Diabetic")
    


# In[ ]:




