import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

df=pd.read_csv('parkinsons.data')

print(df)

print(df.shape)
print(df.isnull().sum())
print(df.info())
print(df.describe())
print(df.columns)

plt.figure(figsize=(10, 6))

df.status.hist()
plt.xlabel('status')
plt.ylabel('Frequencies')
plt.plot()
# The dataset has high number of patients effected with Parkinson's disease.

plt.figure(figsize=(10, 6))
sns.barplot(x="status",y="NHR",data=df);
# The patients effected with Parkinson's disease have high NHR that is the measures of ratio of noise to tonal components in the voice.

plt.figure(figsize=(10, 6))
sns.barplot(x="status",y="HNR",data=df);
# The patients effected with Parkinson's disease have high HNR that is the measures of ratio of noise to tonal components in the voice.

plt.figure(figsize=(10, 6))
sns.barplot(x="status",y="RPDE",data=df);
# The nonlinear dynamical complexity measure RPDE is high in the patients effected with Parkinson's disease.

rows = 3
cols = 7
fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 4))
col = df.columns
index = 1
for i in range(rows):
    for j in range(cols):
        sns.distplot(df[col[index]], ax=ax[i][j])
        index = index + 1

plt.tight_layout()
# Distribution plot
# A distribution plot displays a distribution and range of a set of numeric values plotted against a dimension
df.drop(['name'],axis=1,inplace=True)
# Removing  name column for machine learning algorithms.
X=df.drop(labels=['status'],axis=1)
Y=df['status']
print(X.head())
print(Y.head())
### Spitting the dataset into x and y
plt.show()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=40)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
# Splitting the data into x_train, y_train, x_test, y_test

#Machine learning
#Logistic Regression
log_reg = LogisticRegression().fit(X_train, Y_train)

#predict on train
train_preds = log_reg.predict(X_train)
#accuracy on train
print("Model accuracy on train for LR is: ", accuracy_score(Y_train, train_preds))

#predict on test
test_preds = log_reg.predict(X_test)
#accuracy on test
print("Model accuracy on test for LR is: ", accuracy_score(Y_test, test_preds))
print('-'*50)

# #Confusion matrix
# print("confusion_matrix train is: ", confusion_matrix(Y_train, train_preds))
# print("confusion_matrix test is: ", confusion_matrix(Y_test, test_preds))

#Random Forest
RF=RandomForestClassifier().fit(X_train,Y_train)
#predict on train
train_preds2 = RF.predict(X_train)
#accuracy on train
print("Model accuracy on train for RFC is: ", accuracy_score(Y_train, train_preds2))

#predict on test
test_preds2 = RF.predict(X_test)
#accuracy on test
print("Model accuracy on test for RFC is: ", accuracy_score(Y_test, test_preds2))

# #Confusion matrix
# print("confusion_matrix train is: ", confusion_matrix(Y_train, train_preds2))
# print("confusion_matrix test is: ", confusion_matrix(Y_test, test_preds2))

# # Wrong Predictions made.
# print((Y_test !=test_preds2).sum(),'/',((Y_test == test_preds2).sum()+(Y_test != test_preds2).sum()))
#
# # Kappa Score
# print('KappaScore is: ', metrics.cohen_kappa_score(Y_test,test_preds2))
#
# ## Let us go ahead and compare the predicted and actual values
# print(test_preds2)
#
# print(test_preds2,Y_test)

# ## Saving the actual and predicted values to a dataframe
# ddf=pd.DataFrame(data=[test_preds2,Y_test])
# print(ddf.T)
# 0 means Predicted Value and 1 is True Value.
# # Random forest model gives us an accuracy of 94 percent compared to logistic regression which gave us 84 percent accuracy

#Decision Trees
#fit the model on train data
DT = DecisionTreeClassifier().fit(X,Y)

#predict on train
train_preds3 = DT.predict(X_train)
#accuracy on train
print("Model accuracy on train for DT is: ", accuracy_score(Y_train, train_preds3))

#predict on test
test_preds3 = DT.predict(X_test)
#accuracy on test
print("Model accuracy on test for DT is: ", accuracy_score(Y_test, test_preds3))
# #Confusion matrix
# print("confusion_matrix train is: ", confusion_matrix(Y_train, train_preds3))
# print("confusion_matrix test is: ", confusion_matrix(Y_test, test_preds3))
# print('Wrong predictions out of total')
# print('-'*50)
#
# # Wrong Predictions made.
# print((Y_test !=test_preds3).sum(),'/',((Y_test == test_preds3).sum()+(Y_test != test_preds3).sum()))
# print('-'*50)
#
# # Kappa Score
# print('KappaScore is: ', metrics.cohen_kappa_score(Y_test,test_preds3))
#

#Naive Bayes Classifier
NB=GaussianNB()
NB.fit(X_train,Y_train)
#fit the model on train data
NB=GaussianNB()
NB.fit(X_train,Y_train)

#predict on train
train_preds4 = NB.predict(X_train)
#accuracy on train
print("Model accuracy on train for NB is: ", accuracy_score(Y_train, train_preds4))

#predict on test
test_preds4 = NB.predict(X_test)
#accuracy on test
print("Model accuracy on test for NB is: ", accuracy_score(Y_test, test_preds4))

# #Confusion matrix
# print("confusion_matrix train is: ", confusion_matrix(Y_train, train_preds4))
# print("confusion_matrix test is: ", confusion_matrix(Y_test, test_preds4))
# print('Wrong predictions out of total')
# print('-'*50)
#
# # Wrong Predictions made.
# print((Y_test !=test_preds4).sum(),'/',((Y_test == test_preds4).sum()+(Y_test != test_preds4).sum()))
# print('-'*50)
#
# # Kappa Score
# print('KappaScore is: ', metrics.cohen_kappa_score(Y_test,test_preds4))

# K-NearestNeighbours

#fit the model on train data
KNN = KNeighborsClassifier().fit(X_train,Y_train)
#predict on train
train_preds5 = KNN.predict(X_train)
#accuracy on train
print("Model accuracy on train for KNN is: ", accuracy_score(Y_train, train_preds5))

#predict on test
test_preds5 = KNN.predict(X_test)
#accuracy on test
print("Model accuracy on test for KNN is: ", accuracy_score(Y_test, test_preds5))
# #Confusion matrix
# print("confusion_matrix train is: ", confusion_matrix(Y_train, train_preds5))
# print("confusion_matrix test is: ", confusion_matrix(Y_test, test_preds5))
# print('Wrong predictions out of total')
# print('-'*50)
#
# # Wrong Predictions made.
# print((Y_test !=test_preds5).sum(),'/',((Y_test == test_preds5).sum()+(Y_test != test_preds5).sum()))
#
# print('-'*50)
# # Kappa Score
# print('KappaScore is: ', metrics.cohen_kappa_score(Y_test,test_preds5))

# SupportVectorMachine
#fit the model on train data
SVM = SVC(kernel='linear')
SVM.fit(X_train, Y_train)

#predict on train
train_preds6 = SVM.predict(X_train)
#accuracy on train
print("Model accuracy on train for SVM is: ", accuracy_score(Y_train, train_preds6))

#predict on test
test_preds6 = SVM.predict(X_test)
#accuracy on test
print("Model accuracy on test for SVM is: ", accuracy_score(Y_test, test_preds6))

# #Confusion matrix
# print("confusion_matrix train is: ", confusion_matrix(Y_train, train_preds6))
# print("confusion_matrix test is: ", confusion_matrix(Y_test, test_preds6))
# print('Wrong predictions out of total')
# print('-'*50)
#
# print("recall", metrics.recall_score(Y_test, test_preds6))
# print('-'*50)
#
# # Wrong Predictions made.
# print((Y_test !=test_preds6).sum(),'/',((Y_test == test_preds6).sum()+(Y_test != test_preds6).sum()))
# print('-'*50)
#
# # Kappa Score
# print('KappaScore is: ', metrics.cohen_kappa_score(Y_test,test_preds6))
