#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as nm
data=pd.read_csv("heart.csv")
data.head()


# In[14]:


corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')
# any NA values are automatically excluded
#non-numeric datatypes coloums are ignored 


# In[15]:


cp = pd.get_dummies(data['cp'], prefix = "cp", drop_first=True)
thal = pd.get_dummies(data['thal'], prefix = "thal" , drop_first=True)
slope = pd.get_dummies(data['slope'], prefix = "slope", drop_first=True)
new_data = pd.concat([data, cp, thal, slope], axis=1)
new_data.drop(['cp', 'thal', 'slope'], axis=1, inplace=True)
new_data


# In[16]:


X = new_data.iloc[:,:-1].values
y = new_data.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[17]:


from sklearn.preprocessing import StandardScaler
Sc = StandardScaler()
X_train = Sc.fit_transform(X_train)
X_test = Sc.transform(X_test)


# In[18]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=3)
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)


# In[19]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[20]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred)) 
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[21]:


from sklearn.model_selection import GridSearchCV
import numpy as np
DT = DecisionTreeClassifier(random_state=0)                                      # Instantiate Decision Tree model
parameters = [{'max_depth': np.arange(16), 'min_samples_leaf': np.arange(16), 
               'criterion' : ['gini', 'entropy']}]                                   # Set parameters of the model


grid_search = GridSearchCV(estimator = DT, param_grid = parameters,        
                           cv = 10, n_jobs =  -1)                                  # Instantiate Grid Search model
 
grid_search = grid_search.fit(X_train, y_train)                                     # Train Grid Search model

print("best accuracy is :" , grid_search.best_score_)                               # Display best accuracy

grid_search.best_params_   # best_parms_  is a method in Grid Search to return The Best with resepct to the Metric


# In[11]:


DT = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1,
                            criterion='gini', random_state=0)          # Instantiate Decision Tree model
DT.fit(X_train, y_train)                                                        # Train Random Forest model
print('train accuracy', DT.score(X_train, y_train))                             # Display train accuracy
print('test accuracy', DT.score(X_test, DT.predict(X_test)))                    # Display test accuracy


# In[12]:


#After correlation 
print("Confusion Matrix accuracy:     ",metrics.accuracy_score(y_test, y_pred))
print("Grid Search accuracy :" , grid_search.best_score_) 




# In[ ]:





# In[ ]:





# In[34]:


#Before correlation 
print("Before Grid Search Accuracy:        ",metrics.accuracy_score(y_test, y_pred))
print("After Grid Search Best accuracy is :" , grid_search.best_score_)

