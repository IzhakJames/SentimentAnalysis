from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics

import mlflow

import requests
from bs4 import BeautifulSoup
import pandas as pd
import traceback
from google.cloud import storage

import mysql.connector
from sqlalchemy import create_engine
import pandas as pd
from mysql.connector import Error
from datetime import datetime
from sqlalchemy.types import VARCHAR
import spacy
import streamlit as st
import string
import re
from google.cloud import storage
from io import StringIO

# Initialize an empty DataFrame to store the comments data
comments_data = pd.DataFrame(columns=['Airline','Review ID','Date Published', 'Overall Rating', 'Passenger Country', 'Trip Verified', 'Review Title','Review', 
                                       'Aircraft', 'Type Of Traveller', 'Seat Type', 'Origin', 'Destination', 'Layover', 'Date Flown', 
                                       'Seat Comfort', 'Cabin Staff Service', 'Food & Beverages', 'Ground Service', 
                                       'Value For Money', 'Recommended'])
comments_data_list = [] 

#Functions to webscrape
def fetch_webpage(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch webpage. Status code: {response.status_code}")
        return None

def parse_html(html_content):
    return BeautifulSoup(html_content, 'html.parser')


def get_airline_data(airline_name):
    #Get the maximum pages of airline
    first_url = f"https://www.airlinequality.com/airline-reviews/{airline_name}/?sortby=post_date%3ADesc&pagesize=100"

    #To detect all flight details and subratings
    class_to_label = {
        'aircraft': 'Aircraft',
        'type_of_traveller': 'Type Of Traveller',
        'cabin_flown': 'Seat Type',
        'route': 'Route',
        'date_flown': 'Date Flown',
        'seat_comfort': 'Seat Comfort',
        'cabin_staff_service': 'Cabin Staff Service',
        'food_and_beverages': 'Food & Beverages',
        'inflight_entertainment':'Inflight Entertainment',
        'ground_service': 'Ground Service',
        'wifi_and_connectivity':'Wifi & Connectivity',
        'value_for_money': 'Value For Money',
        'recommended': 'Recommended'
    }
    
    success = False
    html_content = fetch_webpage(first_url)
    for attempt in range(10):
        if html_content:
            soup = parse_html(html_content)

            # Find all comment elements
            comments = soup.find_all('article', itemprop='review')

            for comment in comments:
                try:
                    #Review ID column
                    review_id = comment.get('class')[-1].split("-")[-1]
                    # Date Published column
                    date_published = comment.find('meta', itemprop='datePublished')['content']
                    # Overall Rating column
                    rating_text = comment.find('span', itemprop='ratingValue')
                    rating = rating_text.text if rating_text else ""
                    # Review Title column
                    text_header = comment.find('h2', class_='text_header').text
                    # Passenger Country column
                    text_sub_header_text = comment.find('h3', class_='text_sub_header userStatusWrapper').get_text(strip=True)
                    country = text_sub_header_text.split('(')[-1].split(')')[0]
                    #In older entries, the country isn't listed in the sub header
                    if len(country.split(" ")) > 3:
                        country = ""

                    # Trip Verified and Review
                    text_content = comment.find('div', class_='text_content', itemprop='reviewBody')
                    # Find the element containing 'Not Verified' or 'Trip Verified'
                    verification_text = text_content.find('strong')
                    verification = verification_text.text.strip() if verification_text else ""
                    text_content = text_content.text.strip()
                    #If there is a trip verified before the review
                    if '|' in text_content:
                        text_content= text_content.split('|')[1].strip()

                    # Table that contains all flight details and subratings
                    review_ratings = comment.find('table', class_='review-ratings')
                    review_ratings = comment.find_all('tr')
                    table_data = {}
                    for row in review_ratings:
                        # Find the header and value cells
                        header_cell = row.find('td', class_='review-rating-header')
                        value_cell = row.find('td', class_='review-value')
                        value2_cell = row.find('td', class_='review-rating-stars')
                        #Details of trip
                        if header_cell and value_cell:
                            # Get the class name of the header cell
                            class_name = header_cell['class'][1]
                            # Get the corresponding data label from the class_to_label dictionary
                            data_label = class_to_label.get(class_name, '')
                            value = value_cell.text.strip()
                            # If the feature is 'Route', split the value into origin and destination
                            if data_label == 'Route':
                                origin, destination, layover = "", "", ""
                                # Got layover
                                if ' via ' in value:
                                    layover = value.split(" via ")[1]
                                    value = value.split(" via ")[0]
                                    if " to " in value:
                                        origin, destination = value.split(" to ")
                                    elif " - " in value:
                                        origin, destination = value.split(" - ")
                                elif " then to " in value:
                                    destination = value.split(" then to ")[1]
                                    value = value.split(" then to ")[0]
                                    origin, layover = value.split(" to ")
                                else:
                                    if " to " in value:
                                        origin, destination = value.split(" to ")
                                    elif " - " in value:
                                        origin, destination = value.split(" - ")
                                    layover = ""
                                table_data['Origin'] = origin.strip()
                                table_data['Destination'] = destination.strip()
                                table_data["Layover"] = layover.strip()
                            else:
                                table_data[data_label] = value

                        #Subratings
                        if header_cell and value2_cell:
                            # Get the class name of the header cell
                            class_name = header_cell['class'][1]
                            # Get the corresponding data label from the class_to_label dictionary
                            data_label = class_to_label.get(class_name, '')
                            filled_star_spans = value2_cell.find_all('span', class_='star fill')
                            table_data[data_label] = int(len(filled_star_spans))

                    # Append the data from the current comment to the list
                    comments_data_list.append({'Airline': airline_name, 'Review ID': review_id, 'Date Published': date_published, 'Overall Rating': rating, 
                                                'Passenger Country': country, 'Trip Verified': verification, 
                                                'Review Title': text_header, 'Review': text_content, **table_data})

                except Exception as e:
                    print(f'Error in the comment: -> {comments.index(comment)}')
                    traceback.print_exc()
            success = True
            break
    if not success:
        print(f"Fetching {first_url} failed in 10 attempts")
        return

def webscrapping():
    lst_names = ["scoot","zipair","airasia","cebu-pacific","lion-air","indigo-airlines","jetstar-airways"]
    for name in lst_names:
        get_airline_data(name)
        
    # Convert the list of dictionaries into a DataFrame
    # comments_data = pd.DataFrame(comments_data_list)
    # comments_data.to_csv('all_reviews.csv', encoding='utf-8', index=False)

    df = pd.DataFrame(comments_data_list)
    path_to_private_key = 'dags/ethereal-anthem-417503-1d749c7332fa.json'
    client = storage.Client.from_service_account_json(json_credentials_path=path_to_private_key)
    bucket = client.bucket('bt4301_gp')
    blob = bucket.blob('all_reviews.csv')
    blob.upload_from_string(df.to_csv(), '')

host = '34.87.87.119'
user = 'bt4301_root'
passwd = 'bt4301ftw'
database='bt4301_gp_datawarehouse'
port = '3306'
STOPWORDS = ['i',
 'me',
 'my',
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
 "you're",
 "you've",
 "you'll",
 "you'd",
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 "she's",
 'her',
 'hers',
 'herself',
 'it',
 "it's",
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 "that'll",
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'no',
 'nor',
 'not',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'don',
 "don't",
 'should',
 "should've",
 'now',
 'd',
 'll',
 'm',
 'o',
 're',
 've',
 'y',
 'ain',
 'aren',
 "aren't",
 'couldn',
 "couldn't",
 'didn',
 "didn't",
 'doesn',
 "doesn't",
 'hadn',
 "hadn't",
 'hasn',
 "hasn't",
 'haven',
 "haven't",
 'isn',
 "isn't",
 'ma',
 'mightn',
 "mightn't",
 'mustn',
 "mustn't",
 'needn',
 "needn't",
 'shan',
 "shan't",
 'shouldn',
 "shouldn't",
 'wasn',
 "wasn't",
 'weren',
 "weren't",
 'won',
 "won't",
 'wouldn',
 "wouldn't"]

def remove_punctuation(sentence):
    return ''.join([word for word in str(sentence) if word not in string.punctuation])

def remove_numbers(sentence):
    return re.sub(r'\d+', '', sentence)

def remove_stopwords(sentence):
    return ' '.join([word for word in str(sentence).split() if word not in STOPWORDS])

def remove_spaces(sentence):
    return re.sub(r'\s+', ' ', sentence).strip()

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
def lemmatizer_doc(sentence):
    doc = nlp(sentence)
    new_sentence = [token.lemma_ for token in doc if token.is_alpha]
    return ' '.join(new_sentence)

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

def load_warehouse():
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

    path_to_private_key = 'dags/ethereal-anthem-417503-1d749c7332fa.json'
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

    STOPWORDS.extend(["would", "get", "-", "us", "also", "one", "said", "even", "told", "take", "try", "go", "give", "use", "flight", "airline", "could"])
    STOPWORDS.extend(["cebu","indigo","scoot","airasia","jetstar","lion","zipair","pacific"])

    # 1) Apply lowercase
    reviews_fact['review_cleaned'] = reviews_fact['Review'].apply(lambda text: text.lower())
    # 2) Remove punctuations
    reviews_fact['review_cleaned'] = reviews_fact['review_cleaned'].apply(lambda text: remove_punctuation(text))
    # 3) Remove numbers
    reviews_fact['review_cleaned'] = reviews_fact['review_cleaned'].apply(lambda text: remove_numbers(text))
    # 4) Remove stopwords
    reviews_fact['review_cleaned'] = reviews_fact['review_cleaned'].apply(lambda text: remove_stopwords(text))
    # 5) Remove white spaces
    reviews_fact['review_cleaned'] = reviews_fact['review_cleaned'].apply(lambda text: remove_spaces(text))
    # 6) Apply lemmatization
    
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

import pickle

def predict_data():
    # MODEL_PATH = 'runs:/b9bd1dd88f3c4f1694a48b5b2ca6fd61/mlops_baseline_model_new'
    test_data = pd.read_csv('dags/data/test.csv')

    reviews = test_data['Cleaned_Review'].to_numpy()
    true_labels = test_data['is_negative_sentiment'].to_numpy()

    # mlflow.set_tracking_uri(uri="http://localhost:9080")
    # loaded_model = mlflow.pyfunc.load_model(MODEL_PATH)

    pickle_in = open('dags/python_model.pkl',"rb")
    loaded_model = pickle.load(pickle_in)
    pickle_in.close()

    pred_labels = loaded_model.predict(reviews)
    accuracy = metrics.accuracy_score(true_labels, pred_labels)
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

# webscrapping = PythonOperator(
#     task_id='webscrapping',
#     python_callable=webscrapping,
#     dag=dag
# )

# load_warehouse = PythonOperator(
#     task_id='load_warehouse',
#     python_callable=load_warehouse,
#     dag=dag
# )

predict_data = PythonOperator(
    task_id='predict_data',
    python_callable=predict_data,
    dag=dag
)

# start_task >> webscrapping >> load_warehouse>> predict_data >> end_task
start_task >>  predict_data >> end_task
