from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
import pickle
warnings.filterwarnings("ignore")
labelencoder = LabelEncoder()
log_reg = LogisticRegression()
data = pd.read_csv(r"C:\Users\Jeyanth\Downloads\IBM Attrition Data.csv")
data['Attrition'] = labelencoder.fit_transform(data['Attrition'])
data['Department'] = labelencoder.fit_transform(data['Department'])
data['EducationField'] = labelencoder.fit_transform(data['EducationField'])
data['MaritalStatus'] = labelencoder.fit_transform(data['MaritalStatus'])
x = data.drop(columns=["Attrition"])
x = x.drop(columns=["Department"])
x = x.drop(columns=["EducationField"])
x = x.drop(columns=["MaritalStatus"])
y = data.Attrition
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=.8, random_state=42)
log_reg.fit(x_train, y_train)
pickle.dump(log_reg, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
