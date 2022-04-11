#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as nm
data=pd.read_csv("heart.csv")
data.head()


# In[18]:


corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[19]:


cp = pd.get_dummies(data['cp'], prefix = "cp", drop_first=True)
thal = pd.get_dummies(data['thal'], prefix = "thal" , drop_first=True)
slope = pd.get_dummies(data['slope'], prefix = "slope", drop_first=True)
new_data = pd.concat([data, cp, thal, slope], axis=1)
new_data.drop(['cp', 'thal', 'slope'], axis=1, inplace=True)
new_data


# In[20]:


X = new_data.iloc[:,:-1].values
y = new_data.iloc[:,-1].values
print('shape of X: ',X.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[21]:


from sklearn.preprocessing import StandardScaler
Sc = StandardScaler()
X_train = Sc.fit_transform(X_train)
X_test = Sc.transform(X_test)


# In[22]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
classifier = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)


# In[23]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[24]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[5]:


from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300,400],
    'learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1]}     # Set parameters of the model
AB = AdaBoostClassifier(random_state=0)                     # Instantiate AdaBoost Classifier model
grid_search = GridSearchCV(estimator = AB, param_grid = param_grid, 
                          cv = 10, n_jobs = -1, verbose=2)             # Instantiate Grid Search model
grid_search = grid_search.fit(X_train, y_train)                       # Train Gird Search model
print("best accuracy is :" , grid_search.best_score_)                 # Display best accuracy
grid_search.best_params_    # best_parms_  is a method in Grid Search to return The Best with resepct to the Metric


# In[26]:


AB = AdaBoostClassifier(n_estimators=200 ,
                        learning_rate=0.005, random_state=0)         # Instantiate AdaBoost Classifier model
AB.fit(X_train, y_train)                                             # Train Adaboost Classifier model
print('train accuracy', AB.score(X_train, y_train))                  # Display train accuracy
print('test accuracy ', AB.score(X_test, AB.predict(X_test)))         # display test accuracy


# In[27]:


#After converting the categorical values
print("Before Grid Search Accuracy: ",metrics.accuracy_score(y_test, y_pred))
print("After Grid Search Accuracy:  ",grid_search.best_score_)


# In[ ]:





# In[ ]:





# In[4]:


#Before converting the categorical values
print("Before Grid Search Accuracy: ",metrics.accuracy_score(y_test, y_pred))
print("After Grid Search Accuracy:  ",grid_search.best_score_)


# In[ ]:





# In[ ]:




