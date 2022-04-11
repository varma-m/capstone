#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[19]:


algo = ['Decision', 'Random', 'SVM', 'Logistic','KNN', 'Adaboost', 'XGBoost']
accu=[97.36,94.73,93.42,93.42,92.10,90.78,97.36]
plt.plot(algo,accu)
plt.grid()
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('Confusion Matrix')


# In[21]:


algo = ['Decision', 'Random', 'SVM', 'Logistic','KNN', 'Adaboost', 'XGBoost']
accu=[96.50,93.84,95.63,96.48,90.29,95.59,96.05]
plt.plot(algo,accu)
plt.grid()
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('Grid Search')
#here Decision tree has highest accuracy, next is log with 96.48


# In[22]:


algo = ['Decision', 'Random', 'SVM', 'Logistic','KNN', 'Adaboost', 'XGBoost']
accu=[94.44,91.66,89.18,89.18,86.84,88.57,94.44]
plt.plot(algo,accu)
plt.grid()
plt.xlabel('Algorithm')
plt.ylabel('Precision')
plt.title('Precision')


# In[23]:


algo = ['Decision', 'Random', 'SVM', 'Logistic','KNN', 'Adaboost', 'XGBoost']
accu=[100,97.05,97.05,97.05,97.05,91.17,100]
plt.plot(algo,accu)
plt.grid()
plt.xlabel('Algorithm')
plt.ylabel('Recall')
plt.title('Recall')


# In[ ]:




