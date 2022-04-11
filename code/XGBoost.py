#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as nm
data=pd.read_csv('C:\\Users\\rohit\\OneDrive\\Desktop\\files\\16MIS0299\\code\\heart.csv')
data.head()


# In[2]:


corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')
#feature_cols=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
#X=data[feature_cols]
#y=data.target


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


# In[5]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[6]:


from sklearn.preprocessing import StandardScaler
Sc = StandardScaler()
X_train = Sc.fit_transform(X_train)
X_test = Sc.transform(X_test)


# In[7]:


from xgboost import XGBClassifier
model=XGBClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


# In[8]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[9]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[10]:


from sklearn.model_selection import GridSearchCV
XG = XGBClassifier(random_state=0)                           # Instantiate XG Boost Classifier model
param = {"n_estimators"     : [100, 200,300],
         "learning_rate"    : [0.1, 0.3, 0.5, 0.6],
         "max_depth"        : [2, 3, 4, 5],
         "min_child_weight" : [1, 2, 3, 4],
         "gamma"            : [0.5, 0.8, 0.9, 1],
         "colsample_bytree" : [0.7, 0.8],
         "subsample"        : [0.7, 0.8],
         }                                                        # Set parameters of the model          
grid_search = GridSearchCV(estimator = XG, param_grid = param, 
                          cv = 5, n_jobs = -1, verbose = 2)          # Instantiate Grid Search model
grid_search = grid_search.fit(X_train, y_train)                      # Train Gird Search model

print("best accuracy is :" , grid_search.best_score_)                # Display best accuracy

grid_search.best_params_    # best_parms_  is a method in Grid Search to return The Best with resepct to the Metric


# In[11]:



from sklearn.metrics import plot_confusion_matrix
XGB = XGBClassifier(n_estimators=100, gamma=0.8, learning_rate=0.1,
                      max_depth=5, min_child_weight=1, subsample=0.7,
                      colsample_bytree=0.7, random_state=0)                 # Instantiate XG Boost Classifier model
XGB.fit(X_train, y_train)                                                   # Train XG Boost Classifier model
print('train accuracy', XGB.score(X_train, y_train))                        # Display train accuracy
print('test accuracy', XGB.score(X_test, XGB.predict(X_test)))           # Display test accuracy


# In[12]:


#After correlation 
print("Before Grid Search Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("After Grid Search Accuracy: ",grid_search.best_score_)


# In[ ]:





# In[ ]:





# In[12]:


#Before correlation 
print("Before Grid Search Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("After Grid Search Accuracy: ",grid_search.best_score_)

