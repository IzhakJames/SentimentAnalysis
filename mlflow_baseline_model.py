import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from sklearn.metrics import accuracy_score

import mlflow
from mlflow.models import infer_signature


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

# Process the input text and return sentiment prediction
def is_negative_sentiment_score(text):
    encoded_input = tokenizer(text, truncation=True, return_tensors="pt")  # for PyTorch-based models
    output = model(**encoded_input)
    scores_ = output[0][0].detach().numpy()
    scores_ = softmax(scores_)

    # Format output dictionary of scores
    labels = ["Negative", "Positive"]
    scores = {l: float(s) for (l, s) in zip(labels, scores_)}
    return scores.get("Negative", 0.0)

def predict(test_data):
    th = 0.5    
    test_data['negative_sentiment_score'] = test_data['Cleaned_Review'].apply(is_negative_sentiment_score)
    test_data['sentiment_acc'] = test_data['negative_sentiment_score'].apply(lambda score: 1 if score >= th else 0)

    true_labels = test_data['is_negative_sentiment']
    pred_labels = test_data['sentiment_acc']

    accuracy = accuracy_score(true_labels, pred_labels)
    return accuracy


# init
test_data = pd.read_csv('../data/test.csv')

model_path = "mlops_baseline"
model, tokenizer = load_model(model_path)

test_acc = predict(test_data)


# Set tracking server uri for logging
mlflow.set_tracking_uri(uri="http://localhost:9081")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Tutorial - Baseline Model")

# Start an MLflow run
with mlflow.start_run():

    # Log the hyperparameters
    # mlflow.log_params(params)

    # Log the loss metric, in this case we are using accuracy
    mlflow.log_metric("accuracy", test_acc)

    # Set a tag to identify the experiment run
    mlflow.set_tag("Training Info", "Baseline Model - Sentiment Analysis")

    # Infer the model signature
    signature = infer_signature(test_data, test_acc)

    # Log the model
    model_info = mlflow.pyfunc.log_model(
        python_model=predict,
        artifact_path="mlops_baseline_model",
        pip_requirements=["torch", "transformers", "numpy"],
        signature=signature,
        input_example=test_data,
        registered_model_name="mlops_baseline_model"
    )

    # Note down this model uri to retrieve the model in the future for scoring
    print(model_info.model_uri)
