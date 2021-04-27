import pandas as pd
import csv
from sklearn.tree import DecisionTreeClassifier
#F BOLUMU KULLANICIDAN ALINAN VERININ TAHMINLENMESI
col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
diabetes = pd.read_csv("pima-indians-diabetes.csv", header=None, names=col_names,quoting=csv.QUOTE_NONNUMERIC)
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI','DiabetesPedigreeFunction','Age']

Pregnancies=int(input("Pregnancies  0-17:"))
Glucose=int(input("Glucose  0-199:"))
BloodPressure=int(input("BloodPressure 0-122:"))
SkinThickness=int(input("SkinThickness 0-99:"))
Insulin=int(input("Insulin 0-846:"))
BMI=float(input("BMI 0-67.1:"))
DiabetesPedigreeFunction=float(input("Diabetes Pedigree Function 0.08-2.42:"))
Age=int(input("Age 21-81:"))

list_input=[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
df = pd.DataFrame([list_input])

X = diabetes[feature_cols] # Features
y = diabetes.Outcome # Target variable

clf = DecisionTreeClassifier()
clf = clf.fit(X,y)
y_pred = clf.predict(df)
print("Tahminlenen Sonuç :",y_pred)
