import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import seaborn as sns

df = sns.load_dataset('iris')
df.dropna(inplace=True)
df.dropna(axis=1, inplace=True)
df['species'].unique()  # prints participants

print(df.isnull().sum())  # to check null values in data set
df = df[df['species'] != 'setosa'] #removing versicolor
print(df['species'].unique())
df['species'] = df['species'].map({'versicolor': 0, 'virginica': 1})  #classifying dataset
print(df)

X = df.iloc[:, :-1]
y = df.iloc[: ,-1]

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=2)
classifier = LogisticRegression(solver='saga') # Specify a solver that supports multiple penalties
# initiating hyperparameter tuning...
parameter ={'penalty': ['l2'], 'C': [1,2,3,4,5,10,20], 'max_iter': [1000,2000,3000,4000,5000]}

logreg = GridSearchCV(classifier, param_grid=parameter, scoring='accuracy', cv=5, error_score="raise")
logreg.fit(xtrain, ytrain)

print(logreg.best_params_)
print(logreg.best_score_)

ypred = logreg.predict(xtest)
print(ypred)
from sklearn import metrics
print(metrics.accuracy_score(ytest, ypred))
