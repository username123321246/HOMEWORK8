from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

asd = pd.read_csv("LESSONS\\titanic.csv")

print(asd.head())
print(asd.info())

Y = asd["Survived"]

X = asd[["Pclass", "Sex", "Age", "Fare"]]

X["Sex"] = X["Sex"].replace({"male": 0, "female": 1})
X["Age"] = X["Age"].fillna(X["Age"].median())
X["Fare"] = X["Fare"].fillna(X["Fare"].median())

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.2, random_state = 6)

print(xtrain.shape)
print(ytrain.shape)

model = DecisionTreeClassifier(max_depth = 3, random_state = 23)
model.fit(xtrain, ytrain)

predictions = model.predict(xtest)
print(metrics.accuracy_score(ytest, predictions))




