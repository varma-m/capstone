#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings

data=pd.read_csv("heart.csv")
data.head()


# In[2]:


corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[3]:


cp = pd.get_dummies(data['cp'], prefix = "cp", drop_first=True)
thal = pd.get_dummies(data['thal'], prefix = "thal" , drop_first=True)
slope = pd.get_dummies(data['slope'], prefix = "slope", drop_first=True)
new_data = pd.concat([data, cp, thal, slope], axis=1)
new_data.drop(['cp', 'thal', 'slope'], axis=1, inplace=True)
new_data


# In[4]:


X = new_data.iloc[:,:-1].values
y = new_data.iloc[:,-1].values
print('shape of X: ',X.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[5]:


from sklearn.preprocessing import StandardScaler
Sc = StandardScaler()
X_train = Sc.fit_transform(X_train)
X_test = Sc.transform(X_test)


# In[19]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Instantiate KNN learning model(k=10)
knn = KNeighborsClassifier(n_neighbors=10)

# fit the model
knn.fit(X_train, y_train)


y_pred=knn.predict(X_test)


# In[20]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[21]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[22]:


import numpy as np
from sklearn.model_selection import GridSearchCV
KNC = KNeighborsClassifier()
parameters = [{'n_neighbors': np.arange(16)}]                               # Set parameters of the model


grid_search = GridSearchCV(estimator = KNC, param_grid = parameters,        
                           cv = 5, n_jobs =  -1)                            # Instantiate Grid Search model              
 
grid_search = grid_search.fit(X_train, y_train)                             # Train Grid Search model

print("best accuracy is :" , grid_search.best_score_)                       # Display best accuracy

grid_search.best_params_   # best_parms_  is a method in Grid Search to return The Best with resepct to the Metric


# In[23]:


KNC = KNeighborsClassifier(n_neighbors=8)                               # Instantiate KNN model
KNC.fit(X_train, y_train)                                               # Train KNN model
print('train accuracy', KNC.score(X_train, y_train))                    # Display train accuracy
print('test accuracy', KNC.score(X_test, KNC.predict(X_test)))          # Display test accuracy
grid_search


# In[24]:


#after converting the categorical values
print("Before Grid Search Accuracy:     ",metrics.accuracy_score(y_test, y_pred))
print("After Grid search Best Accuracy is :" , grid_search.best_score_)


# In[ ]:





# In[ ]:





# In[13]:


#before coverting the categorical values 
print("Before Grid Search Accuracy:     ",metrics.accuracy_score(y_test, y_pred))
print("After Grid search Best Accuracy is :" , grid_search.best_score_)

