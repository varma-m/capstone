#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
data=pd.read_csv("heart.csv")
data.head()


# In[13]:


corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[14]:


cp = pd.get_dummies(data['cp'], prefix = "cp", drop_first=True)
thal = pd.get_dummies(data['thal'], prefix = "thal" , drop_first=True)
slope = pd.get_dummies(data['slope'], prefix = "slope", drop_first=True)
new_data = pd.concat([data, cp, thal, slope], axis=1)
new_data.drop(['cp', 'thal', 'slope'], axis=1, inplace=True)
new_data


# In[15]:


X = new_data.iloc[:,:-1].values
y = new_data.iloc[:,-1].values
print('shape of X: ',X.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[16]:


from sklearn.preprocessing import StandardScaler
Sc = StandardScaler()
X_train = Sc.fit_transform(X_train)
X_test = Sc.transform(X_test)


# In[17]:


from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)


y_pred=logreg.predict(X_test)


# In[18]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[19]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[20]:


from sklearn.model_selection import GridSearchCV
log_reg = LogisticRegression(random_state=0) 
parameters = [{"C":[0.07, 0.08, 0.1, 0.3, 0.5, 0.9, 1], "penalty":["l1","l2"]}]     # l1 lasso l2 ridge
grid_search = GridSearchCV(estimator = log_reg, param_grid = parameters,        
                           cv = 10, n_jobs =  -1)                                   # Instantiate Grid Search model
 
grid_search = grid_search.fit(X_train, y_train)                                     # Train Grid Search model

print("best accuracy is :" , grid_search.best_score_)                               # Display best accuracy

grid_search.best_params_   # best_parms_  is a method in Grid Search to return The Best with resepct to the Metric


# In[21]:


from sklearn.metrics import accuracy_score
log_reg = LogisticRegression(C=1, penalty='l2', random_state=0)     # Instantiate Logistic Regression model
log_reg.fit(X_train, y_train)                                       # Train the model
print('Train Accuracy score : ', accuracy_score(y_train, log_reg.predict(X_train))) # Display train accuracy of the model
print('Test Accuracy score : ', accuracy_score(y_test, log_reg.predict(X_test)))  # Display test accuracy of the model


# In[22]:


#After correlation 
print("Before Grid Search Accuracy:     ",metrics.accuracy_score(y_test, y_pred))
print("After Grid search Best Accuracy is :" , grid_search.best_score_)


# In[ ]:





# In[ ]:





# In[11]:


#before correlation 
print("Before Grid Search Accuracy:     ",metrics.accuracy_score(y_test, y_pred))
print("After Grid search Best Accuracy is :" , grid_search.best_score_)

