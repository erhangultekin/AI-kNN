import pandas as pd
from sklearn import metrics, svm
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#IKI FARKLI SINIFLANDIRMA YONTEMIYLE TAHMINLEME D VE E BOLUMU
col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
diabetes = pd.read_csv("pima-indians-diabetes.csv", header=None, names=col_names,quoting=csv.QUOTE_NONNUMERIC)
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI','DiabetesPedigreeFunction','Age']

X = diabetes[feature_cols] # Features
y = diabetes.Outcome # Target variable

def DTC(X,y):
    print("DTC ALGORITMASINA GORE SINIFLANDIRMA VE PRINTLER")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("10 KERE CROSS :",cross_val_score(clf, X, y, cv=10))  # cross value score
    print("Hata Matrisi", cm)  # Confusion
    print("Sonuç:", y_pred)
    print("Karar Ağacı Doğruluk Değeri:", metrics.accuracy_score(y_test, y_pred))  # Accuracy


def kNN(X,y):
    print("KNN ALGORITMASINA GORE SINIFLANDIRMA VE PRINTLER")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("10 KERE CROSS :",cross_val_score(knn, X, y, cv=10))  # cross value score
    print("Hata Matrisi", cm)  # Confusion
    print("Sonuç:", y_pred)
    print("KNN Doğruluk Değeri:", metrics.accuracy_score(y_test, y_pred))  # Accuracy

DTC(X,y)
print("------------------")
kNN(X,y)


