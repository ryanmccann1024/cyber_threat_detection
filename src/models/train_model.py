# Import Libraries
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from joblib import dump, load
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import numpy as np

# Standardize the Data
scaler = StandardScaler()

# Load the Dataset
df = pd.read_csv('2017_Train.csv')

print(df.head())
print(df.shape)

# Visualize Each Digit
feature_colnames = df.columns[:-1]

print(feature_colnames)

# Splitting Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(df[feature_colnames], df['Label'], random_state=0)

X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
X_test = X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

# Fit on training set only.
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression

# multi_class is specifying one versus rest
clf = RandomForestClassifier(max_depth=100, random_state=10)

print("")
print("start time:")
print(datetime.datetime.now())

clf.fit(X_train, y_train)

print("end time:")
print(datetime.datetime.now())

print('Training accuracy:', clf.score(X_train, y_train))
print('Test accuracy:', clf.score(X_test, y_test))

print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), feature_colnames), 
             reverse=True))
print("-----------")


# Save the model...
#dump(clf,'data/model.sav')

