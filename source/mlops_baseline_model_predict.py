import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import roc_auc_score

import mlflow
import mysql.connector
from sqlalchemy import create_engine


def sql_push(data_df):
    host = '34.87.87.119'
    user = 'bt4301_root'
    passwd = 'bt4301ftw'
    database='bt4301_gp_datawarehouse'
    port = '3306'

    db_datawarehouse = mysql.connector.connect(
        host=host,
        user=user,
        passwd=passwd,
        database=database
    )

    cursor = db_datawarehouse.cursor()
    cursor.execute('DROP TABLE IF EXISTS baseline_prediction;')

    engine = create_engine(f'mysql://{user}:{passwd}@{host}:{port}/{database}?charset=utf8mb4', echo=False,future=True)
    db_sent = engine.connect()

    data_df.to_sql(name='baseline_prediction', con=db_sent, if_exists='replace')

    db_sent.commit()
    db_sent.close()


MODEL_PATH = 'runs:/83ccd69c86d848fc930ebea1c53baca6/mlops_baseline_model_new'
mlflow.set_tracking_uri(uri="http://localhost:9080")
loaded_model = mlflow.pyfunc.load_model(MODEL_PATH)

test_data = pd.read_csv('../data/test.csv')

reviews = test_data['Cleaned_Review'].to_numpy()
true_labels = test_data['is_negative_sentiment'].to_numpy()

scores, pred_labels = loaded_model.predict(reviews)
accuracy = metrics.accuracy_score(true_labels, pred_labels)

# Calculate the ROC AUC Score
roc_auc = roc_auc_score(true_labels, scores)
print(f"ROC AUC Score: {roc_auc}")
print(accuracy)

data = {
    "reviews": reviews,
    "is_negative_sentiment": true_labels,
    "scores": scores,
    "sentiment_acc": pred_labels
}

df = pd.DataFrame(data)

sql_push(df)