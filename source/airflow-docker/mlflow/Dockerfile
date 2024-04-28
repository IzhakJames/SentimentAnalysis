# Use the official Python image as a base
FROM python:3.8-slim

# Set the work directory
WORKDIR /mlflow

COPY . /mlflow/

# Install MLflow
RUN pip install mlflow protobuf 

# Expose the port on which MLflow server will run
EXPOSE 9080

# Command to start MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "9080"]
