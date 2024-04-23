import mysql.connector
from sqlalchemy import create_engine
import pandas as pd
from mysql.connector import Error
# import io
# import dropbox
from datetime import datetime
from sqlalchemy.types import VARCHAR
import os
from dotenv import load_dotenv
import nltk 
from nltk.corpus import stopwords
import spacy
import streamlit as st
import string
import re
from google.cloud import storage
from io import StringIO


host = '34.87.87.119'
user = 'bt4301_root'
passwd = 'bt4301ftw'
database='bt4301_gp_datawarehouse'
port = '3306'

def create_database_if_not_exists():
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=passwd 
        )

        cursor = connection.cursor()
        
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor.fetchall()]
        if database in databases:
            print(f"Database '{database}' already exists. No need to create it.")
        else:
            cursor.execute(f"CREATE DATABASE {database}")
            print(f"Database '{database}' created successfully")

    except Error as e:
        print(f"Error: {e}")
    
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

create_database_if_not_exists()

db_datawarehouse = mysql.connector.connect(
	host=host,
	user=user,
	passwd=passwd,
	database=database
)

cursor = db_datawarehouse.cursor()
cursor.execute('DROP TABLE IF EXISTS reviews_fact;')
cursor.execute('DROP TABLE IF EXISTS airline_dimension;')

db_datawarehouse.commit()
db_datawarehouse.close()

engine = create_engine(f'mysql://{user}:{passwd}@{host}:{port}/{database}?charset=utf8mb4', echo=False,future=True)
db_sent = engine.connect()

# load_dotenv()
# token=os.getenv('DROPBOX_TOKEN')
# DBX = dropbox.Dropbox(token)
# _, res = DBX.files_download("/all_reviews.csv")

# with io.BytesIO(res.content) as stream:
#     raw_data = pd.read_csv(stream)

path_to_private_key = 'ethereal-anthem-417503-1d749c7332fa.json'
client = storage.Client.from_service_account_json(json_credentials_path=path_to_private_key)
bucket = client.bucket('bt4301_gp')
blob = bucket.blob('all_reviews.csv')
downloaded_blob = blob.download_as_string()
downloaded_blob = StringIO(downloaded_blob.decode('utf-8'))
raw_data = pd.read_csv(downloaded_blob,index_col=0)


reviews_fact = [
    "Review ID",
    "Date Published",
    "Overall Rating",
    "Passenger Country",
    "Trip Verified",
    "Review",
    "Airline",
    "Type Of Traveller",
    "Seat Type",
    "Seat Comfort",
    "Cabin Staff Service",
    "Food & Beverages",
    "Ground Service",
    "Wifi & Connectivity",
    "Value For Money",
    "Recommended",
    "Inflight Entertainment",
    'Origin',
    'Destination',
    'Layover',
    'Aircraft',
    'Date Flown']

reviews_fact = raw_data[reviews_fact]

reviews_fact['Date Published'] = pd.to_datetime(reviews_fact['Date Published'])

# Add Cleaned Review column
#Stopwords
STOPWORDS = stopwords.words('english')
STOPWORDS.extend(["would", "get", "-", "us", "also", "one", "said", "even", "told", "take", "try", "go", "give", "use", "flight", "airline", "could"])
STOPWORDS.extend(["cebu","indigo","scoot","airasia","jetstar","lion","zipair","pacific"])


#Preprocess texts
# Steps:
# 1) Apply lowercase
# 2) Remove punctuation
# 3) Remove numbers
# 4) Remove stopwords
# 5) Remove white spaces
# 6) Apply lemmatization

# 1) Apply lowercase
reviews_fact['review_cleaned'] = reviews_fact['Review'].apply(lambda text: text.lower())
# 2) Remove punctuations
def remove_punctuation(sentence):
    return ''.join([word for word in str(sentence) if word not in string.punctuation])
reviews_fact['review_cleaned'] = reviews_fact['review_cleaned'].apply(lambda text: remove_punctuation(text))
# 3) Remove numbers
def remove_numbers(sentence):
    return re.sub(r'\d+', '', sentence)
reviews_fact['review_cleaned'] = reviews_fact['review_cleaned'].apply(lambda text: remove_numbers(text))
# 4) Remove stopwords
def remove_stopwords(sentence):
    return ' '.join([word for word in str(sentence).split() if word not in STOPWORDS])
reviews_fact['review_cleaned'] = reviews_fact['review_cleaned'].apply(lambda text: remove_stopwords(text))
# 5) Remove white spaces
def remove_spaces(sentence):
    return re.sub(r'\s+', ' ', sentence).strip()
reviews_fact['review_cleaned'] = reviews_fact['review_cleaned'].apply(lambda text: remove_spaces(text))
# 6) Apply lemmatization
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
def lemmatizer_doc(sentence):
    doc = nlp(sentence)
    new_sentence = [token.lemma_ for token in doc if token.is_alpha]
    return ' '.join(new_sentence)
reviews_fact['review_cleaned'] = reviews_fact['review_cleaned'].apply(lambda text: lemmatizer_doc(text))
st.write('Original review\n', reviews_fact['Review'].iloc[0])
st.write('\nReview clear\n', reviews_fact['review_cleaned'].iloc[0])


reviews_fact.to_sql(name='reviews_fact', con=db_sent, if_exists='replace')

db_sent.commit()

airline_dimension = raw_data[['Airline','Overall Rating']].groupby(by=['Airline']).mean()

sql_types = {
    'Airline': VARCHAR(255)
}

airline_dimension.to_sql(name='airline_dimension', con=db_sent, if_exists='replace',dtype=sql_types)

db_sent.commit()

db_sent.close()

db_sent = mysql.connector.connect(
	host=host,
	user=user,
	passwd=passwd,
	database=database
)
cursor = db_sent.cursor()
cursor.execute('ALTER TABLE airline_dimension MODIFY `Airline` VARCHAR(255);')
cursor.execute('ALTER TABLE reviews_fact MODIFY `Airline` VARCHAR(255);')
cursor.execute('ALTER TABLE reviews_fact ADD PRIMARY KEY (`Review ID`);')
cursor.execute('ALTER TABLE airline_dimension ADD PRIMARY KEY (`Airline`(255))')
cursor.execute('ALTER TABLE reviews_fact ADD FOREIGN KEY (`Airline`) REFERENCES airline_dimension(`Airline`);')

db_sent.commit()
db_sent.close()