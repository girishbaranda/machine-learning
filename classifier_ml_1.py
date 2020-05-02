import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

dataset  = pd.read_csv('iris.csv')

X = dataset.iloc[:, 1:4].values
Y = dataset.iloc[:, 4].values

# imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
# imputer = imputer.fit(X[:, 1:5])
# X[:, 1:5]=imputer.transform(X[:, 1:5])

# le_X = LabelEncoder()
# X[:, 1:5]= le_X.fit_transform(X[:, 1:5])
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)

# oneHE = OneHotEncoder(categorical_features = [1,5])
# X = oneHE.fit_transform(X).toarray()

# sc_X = StandardScaler()
# X = sc_X.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25)

classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

cm = confusion_matrix(Y_test, Y_pred)
print(cm)

target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
print(classification_report(Y_test, Y_pred, target_names=target_names))
