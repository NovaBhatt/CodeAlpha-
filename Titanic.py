#!/usr/bin/env python
# coding: utf-8



import pandas as pd




df = pd.read_csv('./titanic.csv')
print(df)


df.shape



df.describe()




df.info()


# # Data Cleaning


# Filling null cells in "Age" using median values.
df['Age'] = df['Age'].fillna(df['Age'].median())
print(df)




# Dropping "PassengerId", "Name", "Ticket", and "Cabin" since either deemed insignificant or non-numeric.
df = df.loc[:, ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]




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

print(df)



df.info() # no null items found this time! Data Cleaning complete!


# # Testing



# Setting the target class for testing as "Survived".
target = df["Survived"]
print(target)




# Setting features as all other columns.
features = df.drop("Survived", axis = 1)
print(features)



# Preparing the model for testing.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=123)




from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()



model.fit(X_train, y_train)



pred = model.predict(X_test)
print(pred)




from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\


accSc = accuracy_score(y_true = y_test, y_pred = pred)
print(accSc)

preSc = precision_score(y_true = y_test, y_pred = pred)
print(preSc)

recSc = recall_score(y_true = y_test, y_pred = pred)
print(recSc)

f1Sc = f1_score(y_true = y_test, y_pred = pred)
print(f1Sc)
# We see that the f1 score is pretty good. 
# It was found to be better than those for LogisticRegression, BernoulliNB, as well as SVC. 




# Visualizing this graphically through a confusion plot. 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true=y_test, y_pred=pred)
disp = ConfusionMatrixDisplay(cm, display_labels=('True', 'False'))
disp.plot()
# We can see that our model has been trained satisfactorily. 
# It is now time to use it to predict whether under given parameters, a person shall survive, which is the main agenda of the project. 


# # Application


# Making a test class called T_test.
T_test = X_test.loc[:, ['Pclass','Age','SibSp','Parch','Fare','Sex_label','Embarked_C_label','Embarked_Q_label']]


# Defining a dataframe called tf which is assigned all columns of df, except for "Survived" (since that has to be predicted by the model).
tf = df[0:0]
tf= tf.drop('Survived', axis = 1)
print(tf)



# The user can enter the values of the tf columns here. 
tf.loc[len(tf.index)] = ['1', '18','0','2','25.00','0','1','0'] 
print(tf)




# The model predicts the survival status of an individual with user-inputted features. 
pred = model.predict(tf)
print(pred)


# # Answering the second half



# To determine the influence of other factors on the survival of a passenger, the LogisticRegression classifier is the best choice.
from sklearn.linear_model import LogisticRegression

LRmodel = LogisticRegression()




# Visual Python: Machine Learning > Fit/Predict
LRmodel.fit(X_train, y_train)


# In[31]:


# Visual Python: Machine Learning > Fit/Predict
pred = LRmodel.predict(X_test)
print(pred)




# The LRmodel.coef_ attribute contains the coefficients (weights) assigned to each feature in the linear regression model.
# These coefficients represent the impact of each feature on the model’s predictions.
# For example, a positive coef. indicates that an increase in that feature’s value leads to a higher predicted outcome, 
# while a negative coef. suggests the opposite.
LRmodel.coef_



# The LRmodel.feature_names_in_ attribute provides the names of the features used in the linear regression model.
# These feature names correspond to the columns in our dataset.
# In our case, the features include ‘Pclass’, ‘Age’, ‘SibSp’, ‘Parch’, ‘Fare’, ‘Sex_label’, ‘Embarked_C_label’, and ‘Embarked_Q_label’
LRmodel.feature_names_in_


# One can compare the results of the last two blocks and understand which feature has the highest impact on our target, i.e., survival! 
