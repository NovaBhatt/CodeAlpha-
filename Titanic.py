#!/usr/bin/env python
# coding: utf-8

# ## Data import and analysis

# In[273]:


import pandas as pd


# In[274]:


df = pd.read_csv('./titanic.csv')
df


# In[275]:


df.shape


# In[276]:


df.describe()


# In[277]:


df.info()


# ## Data Cleaning

# In[278]:


# Filling null cells in "Age" using median values.
df['Age'] = df['Age'].fillna(df['Age'].median())
df


# In[279]:


# Dropping "PassengerId", "Name", "Ticket", and "Cabin" since either deemed insignificant or non-numeric.
df = df.loc[:, ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]


# In[280]:


# Label encoding "Sex", thus representing "male" as 1 and "female" as 0, and dropping the original column.
df['Sex_label'] = pd.Categorical(df['Sex']).codes
df.drop(['Sex'], axis=1, inplace=True)

# Onehot encoding "Embarked", into "Embarked_C", "Embarked_Q", and "Embarked_S".
df = pd.get_dummies(data=df, columns=['Embarked'])

# Label encoding "Embarked_C" and "Embarked_Q"
df['Embarked_C_label'] = pd.Categorical(df['Embarked_C']).codes
df['Embarked_Q_label'] = pd.Categorical(df['Embarked_Q']).codes

# Dropping "Embarked_C", "Embarked_Q", and "Embarked_S".
df.drop(['Embarked_C'], axis=1, inplace=True)
df.drop(['Embarked_Q'], axis=1, inplace=True)
df.drop(['Embarked_S'], axis=1, inplace=True)

df


# In[281]:


df.info() # no null items found this time! Data Cleaning complete!


# ## Determining the right classifier and testing the program using it. 

# In[282]:


# Setting the target class for testing as "Survived".
target = df["Survived"]
target


# In[283]:


# Setting features as all other columns.
features = df.drop("Survived", axis = 1)
features


# In[284]:


# Preparing the model for testing.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=123)


# ### Logistic Regression classifier

# In[285]:


# To determine the influence of other factors on the survival of a passenger, the LogisticRegression classifier is the best choice.
from sklearn.linear_model import LogisticRegression

LRmodel = LogisticRegression(max_iter=3000) # the (max_iter = 3000) argument helps the program converge better during training. 


# In[286]:


LRmodel.fit(X_train, y_train)


# In[287]:


predLR = LRmodel.predict(X_test)
predLR


# In[288]:


# Visualizing this graphically through a confusion plot. 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
LRcm = confusion_matrix(y_true=y_test, y_pred=predLR)
disp = ConfusionMatrixDisplay(LRcm, display_labels=('True', 'False'))
disp.plot()
# We can see that our model has been trained satisfactorily. 
# It is now time to use it to predict whether under given parameters, a person shall survive, which is the main agenda of the project. 


# In[289]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[290]:


accScLR = accuracy_score(y_true = y_test, y_pred = predLR)
print(accScLR)


# In[291]:


preScLR = precision_score(y_true = y_test, y_pred = predLR)
print(preScLR)


# In[292]:


recScLR = recall_score(y_true = y_test, y_pred = predLR)
print(recScLR)


# In[293]:


f1ScLR = f1_score(y_true = y_test, y_pred = predLR)
f1ScLR


# ### Bernoulli classifier

# In[294]:


from sklearn.naive_bayes import BernoulliNB

modelNB = BernoulliNB()


# In[295]:


modelNB.fit(X_train, y_train)


# In[296]:


predNB = modelNB.predict(X_test)
predNB


# In[297]:


cmNB = confusion_matrix(y_true=y_test, y_pred=predNB)
disp = ConfusionMatrixDisplay(cmNB, display_labels=('True', 'False'))
disp.plot()


# In[298]:


accScNB = accuracy_score(y_true = y_test, y_pred = predNB)
print(accScNB)


# In[299]:


preScNB = precision_score(y_true = y_test, y_pred = predNB)
print(preScNB)


# In[300]:


recScNB = recall_score(y_true = y_test, y_pred = predNB)
print(recScNB)


# In[301]:


f1ScNB = f1_score(y_true = y_test, y_pred = predNB)
f1ScNB


# ### SVC

# In[302]:


from sklearn.svm import SVC

modelSVC = SVC()


# In[303]:


modelSVC.fit(X_train, y_train)


# In[304]:


predSVC = modelSVC.predict(X_test)
predSVC


# In[305]:


cmSVC = confusion_matrix(y_true=y_test, y_pred=predSVC)
disp = ConfusionMatrixDisplay(cmSVC, display_labels=('True', 'False'))
disp.plot()


# In[306]:


accScSVC = accuracy_score(y_true = y_test, y_pred = predSVC)
print(accScSVC)


# In[307]:


preScSVC = precision_score(y_true = y_test, y_pred = predSVC)
print(preScSVC)


# In[308]:


recScSVC = recall_score(y_true = y_test, y_pred = predSVC)
print(recScSVC)


# In[309]:


f1ScSVC = f1_score(y_true = y_test, y_pred = predSVC)
f1ScSVC


# ### Decision Tree classifier

# In[310]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()


# In[311]:


model.fit(X_train, y_train)


# In[312]:


pred = model.predict(X_test)
pred


# In[313]:


cm = confusion_matrix(y_true=y_test, y_pred=pred)
disp = ConfusionMatrixDisplay(cm, display_labels=('True', 'False'))
disp.plot()


# In[314]:


accSc = accuracy_score(y_true = y_test, y_pred = pred)
print(accSc)


# In[315]:


preSc = precision_score(y_true = y_test, y_pred = pred)
print(preSc)


# In[316]:


recSc = recall_score(y_true = y_test, y_pred = pred)
print(recSc)


# In[317]:


f1Sc = f1_score(y_true = y_test, y_pred = pred)
f1Sc
# We see that the f1 score is better than those of LogisticRegression, BernoulliNB, as well as SVC.
# Thus, we shall fo ahead with the Decision Tree classifier. 


# ## Application

# In[318]:


# Making a test class called T_test.
T_test = X_test.loc[:, ['Pclass','Age','SibSp','Parch','Fare','Sex_label','Embarked_C_label','Embarked_Q_label']]


# In[319]:


# Defining a dataframe called tf which is assigned all columns of df, except for "Survived" (since that has to be predicted by the model).
tf = df[0:0]
tf= tf.drop('Survived', axis = 1)
tf


# In[327]:


# The user can enter the values of the tf columns here. 
tf.loc[len(tf.index)] = ['0', '34','1','3','12.00','1','0','1'] 
tf


# In[328]:


# The model predicts the survival status of an individual with user-inputted features. 
pred = model.predict(tf)
pred


# ## Answering the second half

# In[325]:


# The LRmodel.coef_ attribute contains the coefficients (weights) assigned to each feature in the linear regression model.
# These coefficients represent the impact of each feature on the model’s predictions.
# For example, a positive coef. indicates that an increase in that feature’s value leads to a higher predicted outcome, while a negative coef. suggests the opposite.
LRmodel.coef_


# In[326]:


# The LRmodel.feature_names_in_ attribute provides the names of the features used in the linear regression model.
# These feature names correspond to the columns in our dataset.
# In our case, the features include ‘Pclass’, ‘Age’, ‘SibSp’, ‘Parch’, ‘Fare’, ‘Sex_label’, ‘Embarked_C_label’, and ‘Embarked_Q_label’
LRmodel.feature_names_in_


# One can compare the results of the last two blocks and understand which feature has the highest impact on our target, i.e., survival! 
