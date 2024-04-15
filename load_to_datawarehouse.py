import mysql.connector
from sqlalchemy import create_engine
import pandas as pd
from mysql.connector import Error
import io
import dropbox
from datetime import datetime
from sqlalchemy.types import VARCHAR
import os
from dotenv import load_dotenv


host = '34.87.87.119'
user = 'bt4301_root'
passwd = 'bt4301ftw'
database='bt4301_gp_datawarehouse'
port = '3306'

def create_database_if_not_exists():
    try:
        # Connect to the MySQL server
        connection = mysql.connector.connect(
            host=host,
            user=user,  # Replace with your MySQL username
            password=passwd  # Replace with your MySQL password
        )

        # Create a cursor object
        cursor = connection.cursor()
        
        # Check if the database exists
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor.fetchall()]
        if database in databases:
            print(f"Database '{database}' already exists. No need to create it.")
        else:
            # Execute a query to create a database
            cursor.execute(f"CREATE DATABASE {database}")
            print(f"Database '{database}' created successfully")

    except Error as e:
        print(f"Error: {e}")
    
    finally:
        # Close the connection and cursor
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

# Call the function
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
cursor.execute('DROP TABLE IF EXISTS flight_dimension;')


db_datawarehouse.commit()
db_datawarehouse.close()

engine = create_engine(f'mysql://{user}:{passwd}@{host}:{port}/{database}?charset=utf8mb4', echo=False,future=True)
db_sent = engine.connect()

load_dotenv()
token=os.getenv('DROPBOX_TOKEN')
DBX = dropbox.Dropbox(token)
_, res = DBX.files_download("/all_reviews.csv")

with io.BytesIO(res.content) as stream:
    raw_data = pd.read_csv(stream)

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

reviews_fact.to_sql(name='reviews_fact', con=db_sent, if_exists='replace')

db_sent.commit()

airline_dimension = raw_data[['Airline','Overall Rating']].groupby(by=['Airline']).mean()

sql_types = {
    'Airline': VARCHAR(255)  # or another length that suits your data
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