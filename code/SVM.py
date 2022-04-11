#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as nm
data=pd.read_csv(r'D:\SEM 5_2\project\heart.csv')
data.head()


# In[16]:


corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[17]:


cp = pd.get_dummies(data['cp'], prefix = "cp", drop_first=True)
thal = pd.get_dummies(data['thal'], prefix = "thal" , drop_first=True)
slope = pd.get_dummies(data['slope'], prefix = "slope", drop_first=True)
new_data = pd.concat([data, cp, thal, slope], axis=1)
new_data.drop(['cp', 'thal', 'slope'], axis=1, inplace=True)
new_data


# In[18]:


X = new_data.iloc[:,:-1].values
y = new_data.iloc[:,-1].values


# In[19]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[20]:


from sklearn.preprocessing import StandardScaler
Sc = StandardScaler()
X_train = Sc.fit_transform(X_train)
X_test = Sc.transform(X_test)


# In[21]:


from sklearn.svm import SVC # "Support vector classifier"  
classifier = SVC(kernel='linear', random_state=0)  
classifier.fit(X_train, y_train)  
y_pred=classifier.predict(X_test)


# In[22]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[23]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[24]:


import numpy as np
from sklearn.model_selection import GridSearchCV
parameters = [{'C': np.arange(16), 'kernel': ['linear', 'rbf'],
               'gamma': [0.05, 0.1, 0.2, 0.3, 0.5, 1]}]                   # Set parameters of the model
SVM_cl = SVC(random_state=0)                                              # Instantiate SVM model     
grid_search = GridSearchCV(estimator = SVM_cl, param_grid = parameters,        
                           cv = 10, n_jobs =  -1)                           # Instantiate Grid Search model
 
grid_search = grid_search.fit(X_train, y_train)                                     # Train Grid Search model

print("best accuracy is :" , grid_search.best_score_)                               # Display best accuracy

grid_search.best_params_ # best_parms_  is a method in Grid Search to return The Best with resepct to the Metric


# In[25]:


SVM_cl = SVC(C=1, kernel='linear', gamma=0.05, random_state=0)              # Instantiate SVM model 
SVM_cl.fit(X_train, y_train)                                                # Train SVM model
print('train accuracy', SVM_cl.score(X_train, y_train))                     # Display train accuracy
print('test accuracy', SVM_cl.score(X_test, SVM_cl.predict(X_test)))        # Display test accuracy
 


# In[26]:



print("Before Grid Search Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("After Grid Search Accuracy: ",grid_search.best_score_)


# In[ ]:





# In[ ]:





# In[14]:


#Before
print("Before Grid Search Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("After Grid Search Accuracy: ",grid_search.best_score_)

