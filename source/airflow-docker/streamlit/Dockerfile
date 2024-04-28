# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

# Expose the port the app runs on
EXPOSE 8501

# Define the command to run your app using streamlit command
CMD ["streamlit", "run", "Streamlit.py"]