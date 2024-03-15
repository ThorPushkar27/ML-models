#!/usr/bin/env python
# coding: utf-8

# In[71]:


# Importing the dependencies.
import numpy as np
import pandas as pd
import statistics


# # K-Nearest Neighbors Classifier

# In[68]:


# Known as Lazy Model.
class KNN_Classifier():
    
    # Initiating the parameters.
    def __init__(self, distance_metric):
        self.distance_metric = distance_metric
       
    # Getting the distance measure
    def get_distance_metric(self, training_data_point, test_data_point):
        if(self.distance_metric == "euclidean"):
            dist =0
            for i in range(len(training_data_point)-1): # Excluding the last target feature.
                dist = dist + (training_data_point[i]- test_data_point[i]) ** 2
                
            euclidean_dist = np.sqrt(dist)
            return euclidean_dist
        
        elif (self.distance_metric == "manhattan"):
            dist =0
            for i in range(len(training_data_point)-1): # Excluding the last target feature.
                dist = dist + abs(training_data_point[i]- test_data_point[i])
            
            manhattan_dist = dist
            return manhattan_dist
        
        
     
    # Getting the nearest neighbors
    def nearest_neighbors(self,X_train, test_data, k):
        distance_list = []
        for training_data in X_train:
            distance = self.get_distance_metric(training_data,test_data)
            distance_list.append((training_data, distance)) 
       
        distance_list.sort(key= lambda x:x[1])
        
        #finding the k neighbors 
        neighbors_list = []
            
        for j in range(k):
            neighbors_list.append(distance_list[j][0])
        return neighbors_list

    # Predicting the class of new data points.       
    def predict(self, X_train, test_data, k):
        neighbors = self.nearest_neighbors(X_train,test_data,k)
        
        for data in neighbors:
            label=[]
            label.append(data[-1])
            
        predicted_class = statistics.mode(label)
        
        return predicted_class
            
    


#  Diabetes Prediction

# In[87]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# In[7]:


diabetes = pd.read_csv("diabetes.csv")

diabetes.head()


# In[8]:


diabetes.shape


# In[10]:



X = diabetes.drop(columns="Outcome", axis = 1)
Y = diabetes["Outcome"]


# In[15]:


# Converting the data to the numpy array
X = X.to_numpy()
Y = Y.to_numpy()


# In[18]:


print(X)


# In[19]:


print(Y)


# In[20]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify = Y, random_state=2)


# In[21]:


print(X.shape, X_train.shape, X_test.shape)


# In[22]:


X_train


# In[26]:


# Inserting Y_train values in the X_train array.
X_train = np.insert(X_train, 8 , Y_train, axis=1)


# In[46]:


np.delete(X_train,9,1)


# In[52]:


np.delete(X_train,9,1)


# In[55]:


X_train_del = np.delete(X_train,10,1)


# In[62]:


X_train_del1 = np.delete(X_train_del,9,1)


# In[63]:


X_train_del1.shape


# In[30]:


print(X_train[:,8])


# In[54]:


X_train.shape


# X_train --> training data with the features and target.
# X_test ---> test data without target

# Model Training: KNN Classifier

# In[69]:


classifier = KNN_Classifier(distance_metric='euclidean')


# NOTE: The KNN Classifier can predict the label for only one data point at a time.

# In[76]:


prediction = classifier.predict(X_train_del1, X_test[5], k=5)


# In[77]:


print(X_test[5])


# In[78]:


print(Y_test[5])


# In[79]:


print(prediction)


# In[80]:


X_test.shape


# In[81]:


X_test_size = X_test.shape[0]
X_test_size


# In[82]:


y_pred = []
for i in range(X_test_size):
    prediction = classifier.predict(X_train_del1, X_test[i], k = 5)
    y_pred.append(prediction)
print(y_pred)


# In[83]:


y_true = Y_test
print(y_true)


# Model_Evaluation

# In[84]:


accuracy = accuracy_score(y_pred, y_true)


# In[86]:


print(accuracy*100)


# Using KNeighborsClassifier in SKLEARN Library

# In[89]:


classifier = KNeighborsClassifier(p=2) #p=2 for Euclidean metric.


# In[95]:


X_train_del2 = np.delete(X_train_del1,8,1)


# In[98]:


X_train_del2.shape


# In[97]:


classifier.fit(X_train_del2, Y_train)


# In[92]:


X_test.shape


# In[93]:


X_train_del1.shape


# In[99]:


y_pred = classifier.predict(X_test)


# In[100]:


accuracy = accuracy_score(y_pred,Y_test)


# In[101]:


print("The accuracy score in Percentage is:", accuracy*100)


# In[ ]:




