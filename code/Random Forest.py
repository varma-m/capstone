#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as nm
df=pd.read_csv(r'D:\SEM 5_2\project\heart.csv')
df.head()


# In[48]:


corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[49]:


cp = pd.get_dummies(data['cp'], prefix = "cp", drop_first=True)
thal = pd.get_dummies(data['thal'], prefix = "thal" , drop_first=True)
slope = pd.get_dummies(data['slope'], prefix = "slope", drop_first=True)
new_data = pd.concat([data, cp, thal, slope], axis=1)
new_data.drop(['cp', 'thal', 'slope'], axis=1, inplace=True)
new_data


# In[50]:


X = new_data.iloc[:,:-1].values
y = new_data.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[51]:


from sklearn.preprocessing import StandardScaler
Sc = StandardScaler()
X_train = Sc.fit_transform(X_train)
X_test = Sc.transform(X_test)


# In[52]:


from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


# In[53]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[54]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[55]:


from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth': [14, 15, 17],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [4, 5, 6],
    'n_estimators': [200, 300, 500]}                     # Set parameters of the model
RF = RandomForestClassifier(random_state=0)                             # Instantiate Random Forest model
grid_search = GridSearchCV(estimator = RF, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 1)          # Instantiate Grid Search model
grid_search = grid_search.fit(X_train, y_train)                      # Train Grid Search model

print("best accuracy is :" , grid_search.best_score_)                # Display best accuracy

grid_search.best_params_    # best_parms_  is a method in Grid Search to return The Best with resepct to the Metric


# In[56]:


RF_clf = RandomForestClassifier(n_estimators=200, max_depth=14,
                                min_samples_leaf=3, min_samples_split=4,
                                random_state=0)                             # Instantiate Random forest model
RF_clf.fit(X_train, y_train)                                                # Train Random Forest model
print('train accuracy', RF_clf.score(X_train, y_train))                     # Display train accuracy
print('test accuracy', RF_clf.score(X_test, RF_clf.predict(X_test)))        # Display test accuracy
      


# In[57]:


#After correlation 
print("Before Grid Search Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("After Grid Search Accuracy :" , grid_search.best_score_)


# In[ ]:





# In[ ]:





# In[46]:


#before correlation 
print("Before Grid Search Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("After Grid Search Accuracy :" , grid_search.best_score_)

