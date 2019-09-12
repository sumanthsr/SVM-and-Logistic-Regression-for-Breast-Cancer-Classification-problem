
# coding: utf-8

# ## Importing necessary modules

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing the breast cancer data from scikit-learn's datasets module 

# In[2]:


from sklearn.datasets import load_breast_cancer
c_df = load_breast_cancer()


# ## Creating dataframe from imported data

# In[3]:


bc_df = pd.DataFrame(np.c_[c_df['data'], c_df['target']], columns = np.append(c_df['feature_names'], ['target']))
bc_df.head()


# In[4]:


bc_df['target'].value_counts()


# In[5]:


sns.pairplot(bc_df, hue = 'target', vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness','mean concavity', 'mean symmetry'])


# ## Correlation Matrix

# In[6]:


plt.figure(figsize = (20,20))
sns.heatmap(bc_df.corr(), annot = True)


# ## Splitting data into train and test sets

# In[7]:


from sklearn.model_selection import train_test_split
#defining X set 
X = bc_df.drop(['target'], axis = 1)
print(X.head(n=20))
y = bc_df['target']
print(y.head(n=20))

#training and testing set definition
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ## SVM Classifier

# In[8]:


from sklearn.svm import SVC

#Initializing model
svector_model = SVC()

#fitting model
svector_model.fit(X_train, y_train)

#predicting outcomes with the fit model
y_pred = svector_model.predict(X_test)

#importing modules for confusion matrix and class report
from sklearn.metrics import confusion_matrix, classification_report

conf_matrix = np.array(confusion_matrix(y_test, y_pred, labels = [1,0]))

confm_df = pd.DataFrame(conf_matrix, index = ['Malignant', 'Benign'], columns = ['Pred_Malignant', 'Pred_Benign'])

sns.heatmap(confm_df, annot = True)


# In[9]:


#classification report 
clsfn_report = classification_report(y_test, y_pred)
print(clsfn_report)


# ## Improving Model for better classification outcome

# In[10]:


#min and max factors
X_train_min = X_train.min()
X_train_max = X_train.max()

#defining range
X_train_range = (X_train_max - X_train_min)

#scaling training set
X_train_scaled = (X_train - X_train_min)/(X_train_range)

print('X_train min, max, range and scaled \n')
print(X_train_min)
print('\n')
print(X_train_max)
print('\n')
print(X_train_range)
print('\n')
print(X_train_scaled.head())


# ## Normalizing the training dataset  

# In[11]:


#scaling test dataset
X_test_min = X_test.min()

X_test_range = (X_test - X_test_min).max()

X_test_scaled = (X_test - X_test_min)/X_test_range


# # Retraining the SVM Classifier with scaled train and test datasets

# In[12]:


#initializing the model
svscaled_model = SVC()

#fitting the model
svscaled_model.fit(X_train_scaled, y_train)

#predicting outcomes using the fit model
y_pred = svscaled_model.predict(X_test_scaled)


# ## Confusion Matrix for prediction by retrained SVC model

# In[13]:


cmatrix = confusion_matrix(y_test, y_pred)

cmatrix = np.array(confusion_matrix(y_test, y_pred, labels = [1,0]))

cmatrix = pd.DataFrame(cmatrix, index = ['Malignant', 'Benign'], columns = ['Pred_Malignant', 'Pred_Benign'])

sns.heatmap(cmatrix, annot = True)


# In[14]:


#classification report 
print(classification_report(y_test, y_pred))


# ## Logistic Regression

# In[15]:


#importing logreg model
from sklearn.linear_model import LogisticRegression

#initializing the logreg model
logreg = LogisticRegression()

#fitting the logreg model
logreg.fit(X_train_scaled, y_train)

#predicting using fit model
y_pred = logreg.predict(X_test_scaled)

print('The accuracy of logreg model:{:.2f}'.format(logreg.score(X_test, y_test)))

cmtx = confusion_matrix(y_test, y_pred)

print(cmtx)

cmtx = np.array(confusion_matrix(y_test, y_pred, labels = [1,0]))

cmtx = pd.DataFrame(cmtx, index = ['Malignant', 'Benign'], columns = ['Pred_Malignant', 'Pred_Benign'])

sns.heatmap(cmtx, annot = True)

print(classification_report(y_test, y_pred))


# ### Re-Splitting data with 30% test size and random_state equal to 0

# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('Accuracy of logreg model after re-splitting and at random state 0:{:.2f}'.format(logreg.score(X_test, y_test)))

cmtx = confusion_matrix(y_test, y_pred)

print(cmtx)

cmtx = np.array(confusion_matrix(y_test, y_pred, labels = [1,0]))

cmtx = pd.DataFrame(cmtx, index = ['Malignant', 'Benign'], columns = ['Pred_Malignant', 'Pred_Benign'])

sns.heatmap(cmtx, annot = True)

cls_report = classification_report(y_test, y_pred)

print(cls_report)

