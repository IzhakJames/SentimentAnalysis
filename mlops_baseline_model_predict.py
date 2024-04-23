import pandas as pd

from sklearn import metrics

import mlflow
# from mlflow.models import infer_signature

MODEL_PATH = 'runs:/ff966c0493704becafdce737630a9aa5/mlops_baseline_model_new'
test_data = pd.read_csv('../data/test.csv')

reviews = test_data['Cleaned_Review'].to_numpy()
true_labels = test_data['is_negative_sentiment'].to_numpy()

mlflow.set_tracking_uri(uri="http://localhost:9080")
loaded_model = mlflow.pyfunc.load_model(MODEL_PATH)

pred_labels = loaded_model.predict(reviews)
accuracy = metrics.accuracy_score(true_labels, pred_labels)
print(accuracy)
