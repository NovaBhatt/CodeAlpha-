#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('./titanic.csv')
df


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.info()


# # Data Cleaning

# In[7]:


# Filling null cells in "Age" using median values.
df['Age'] = df['Age'].fillna(df['Age'].median())
df


# In[8]:


# Dropping "PassengerId", "Name", "Ticket", and "Cabin" since either deemed insignificant or non-numeric.
df = df.loc[:, ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]


# In[9]:


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


# In[10]:


df.info() # no null items found this time! Data Cleaning complete!


# # Testing

# In[11]:


# Setting the target class for testing as "Survived".
target = df["Survived"]
target


# In[12]:


# Setting features as all other columns.
features = df.drop("Survived", axis = 1)
features


# In[13]:


# Preparing the model for testing.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=123)


# In[14]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()


# In[15]:


model.fit(X_train, y_train)


# In[16]:


pred = model.predict(X_test)
pred


# In[18]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[19]:


f1Sc = f1_score(y_true = y_test, y_pred = pred)
f1Sc
# We see that the f1 score is pretty good. 
# It was found to be better than those for LogisticRegression, BernoulliNB, as well as SVC. 


# In[20]:


# Visualizing this graphically through a confusion plot. 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true=y_test, y_pred=pred)
disp = ConfusionMatrixDisplay(cm, display_labels=('True', 'False'))
disp.plot()
# We can see that our model has been trained satisfactorily. 
# It is now time to use it to predict whether under given parameters, a person shall survive, which is the main agenda of the project. 


# # Application

# In[22]:


# Making a test class called T_test.
T_test = X_test.loc[:, ['Pclass','Age','SibSp','Parch','Fare','Sex_label','Embarked_C_label','Embarked_Q_label']]


# In[24]:


# Defining a dataframe called tf which is assigned all columns of df, except for "Survived" (since that has to be predicted by the model).
tf = df[0:0]
tf= tf.drop('Survived', axis = 1)
tf


# In[27]:


# The user can enter the values of the tf columns here. 
tf.loc[len(tf.index)] = ['1', '18','0','2','25.00','0','1','0'] 
tf


# In[28]:


# The model predicts the survival status of an individual with user-inputted features. 
pred = model.predict(tf)
pred


# # Answering the second half

# In[29]:


# To determine the influence of other factors on the survival of a passenger, the LogisticRegression classifier is the best choice.
from sklearn.linear_model import LogisticRegression

LRmodel = LogisticRegression()


# In[30]:


# Visual Python: Machine Learning > Fit/Predict
LRmodel.fit(X_train, y_train)


# In[31]:


# Visual Python: Machine Learning > Fit/Predict
pred = LRmodel.predict(X_test)
pred


# In[35]:


# The LRmodel.coef_ attribute contains the coefficients (weights) assigned to each feature in the linear regression model.
# These coefficients represent the impact of each feature on the model’s predictions.
# For example, a positive coef. indicates that an increase in that feature’s value leads to a higher predicted outcome, 
# while a negative coef. suggests the opposite.
LRmodel.coef_


# In[36]:


# The LRmodel.feature_names_in_ attribute provides the names of the features used in the linear regression model.
# These feature names correspond to the columns in our dataset.
# In our case, the features include ‘Pclass’, ‘Age’, ‘SibSp’, ‘Parch’, ‘Fare’, ‘Sex_label’, ‘Embarked_C_label’, and ‘Embarked_Q_label’
LRmodel.feature_names_in_


# One can compare the results of the last two blocks and understand which feature has the highest impact on our target, i.e., survival! 
