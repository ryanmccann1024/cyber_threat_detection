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

clf=load('data/model.sav')

# Display classes
print(clf.classes_)

# Load sample from file
# 
sample_file=pd.read_csv('2017_Train_DDOS.csv')
sample_colnames=sample_file.columns[:-1]
sample_file = sample_file.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
sample_test=scaler.transform(sample_file[sample_colnames])

prediction=clf.predict(sample_test)
print(prediction)
pd.DataFrame(prediction, columns=['predictions']).to_csv('predict_2017_Train_DDOS.csv')


print('Done.')
