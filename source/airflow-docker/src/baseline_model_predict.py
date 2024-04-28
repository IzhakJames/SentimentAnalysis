import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics

import mlflow
from mlflow.models import infer_signature



df = pd.read_csv('../data/iris.csv')

X = pd.concat([df['sepal_length'], df['sepal_width'], df['petal_length'], df['petal_width']], axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, shuffle = True, stratify = y)

mlflow.set_tracking_uri(uri="http://localhost:9080")
loaded_model = mlflow.pyfunc.load_model('runs:/0ad62ab3a0754013a65ce4bc703eed9a/baseline_model')
y_test_pred = loaded_model.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_test_pred)

print(accuracy)
