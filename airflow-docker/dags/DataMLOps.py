from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
import pickle

import mlflow
from mlflow.models import infer_signature

def print_hello():
    df = pd.read_csv('dags/data/iris.csv')

    X = pd.concat([df['sepal_length'], df['sepal_width'], df['petal_length'], df['petal_width']], axis=1)
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, shuffle = True, stratify = y)

    print("Loading URI")
    mlflow.set_tracking_uri(uri="http://localhost:9080")

    print("Finish Loading URI, loading model")
    loaded_model = mlflow.pyfunc.load_model('runs:/0ad62ab3a0754013a65ce4bc703eed9a/baseline_model')


    print("Predicting now")
    y_test_pred = loaded_model.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_test_pred)

    print(accuracy)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime.now(),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'daily_task_dag',
    default_args=default_args,
    description='A simple DAG that runs daily at 8am',
    schedule_interval='0 8 * * *',
)

start_task = DummyOperator(task_id='start_task', dag=dag)
end_task = DummyOperator(task_id='end_task', dag=dag)
task_2 = PythonOperator(
    task_id='task_2',
    python_callable=print_hello,
    dag=dag
)

start_task >> task_2 >> end_task
