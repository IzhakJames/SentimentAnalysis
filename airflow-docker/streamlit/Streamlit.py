import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# EDA
import mysql.connector
from sqlalchemy import create_engine
from mysql.connector import Error
import collections
import seaborn as sns
from wordcloud import WordCloud 

# ML
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay

st.set_option('deprecation.showPyplotGlobalUse', False)

# Connect to MySQL database
def connect_to_database():
    try:
        connection = mysql.connector.connect(
            host='34.87.87.119',
            user='bt4301_root',
            password='bt4301ftw',
            database='bt4301_gp_datawarehouse',
            port='3306'
        )
        if connection.is_connected():
            db_Info = connection.get_server_info()
            #st.write("Connected to MySQL Server version ", db_Info)
            return connection
    except Error as e:
        st.error(f"Error connecting to MySQL database: {e}")

# Fetch data from MySQL database
def fetch_data(connection, table):
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute(f"SELECT * FROM {table}")
            records = cursor.fetchall()
            columns = [col[0] for col in cursor.description]  # Fetch column names
            return records, columns
        except mysql.connector.Error as e:
            st.error(f"Error fetching data from MySQL database: {e}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
                #st.write("MySQL connection is closed")

# Get dataframesl from SQL
def get_dataframe(col):
    # Connect to the database
    connection = connect_to_database()

    # Fetch and display data
    if connection:
        data, columns = fetch_data(connection, col)
        if data:
            #st.write("Data from MySQL Database:")
            df = pd.DataFrame(data, columns=columns)  # Create DataFrame with fetched data and columns
            df = df.reset_index(drop=True)  # Reset index
            return df
        
def single_column_analysis(data, col, title, horizontal, top_10):
    unique_counts = data[col].value_counts().sort_values(ascending = True)
    if top_10:
        unique_counts = unique_counts.tail(10)
    if horizontal:
        plt.barh(unique_counts.index, unique_counts.values)
    else:
        plt.bar(unique_counts.index, unique_counts.values)
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.title(f'Count of Each {title}')
    st.pyplot() 

def create_corpus(column,df):
    corpus = []
    
    for sentence in df[column].str.split():
        for word in sentence:
            corpus.append(word.lower())
    
    return corpus

def airline_wordcloud(airline):
    all_words = ' '.join(create_corpus("review_cleaned", reviews_fact[reviews_fact["Airline"] == airline]))
    wordcloud = WordCloud(max_font_size=50, background_color='white').generate(all_words)
    plt.figure(figsize=(15,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Cleaned {airline} reviews wordcloud")
    st.pyplot()

# Start of frontend
# Create a horizontal navigation bar
nav_option = st.sidebar.radio("Navigation Bar", ["Home", "Exploratory Data Analysis", "ML"])

# Display content based on navigation selection
if nav_option == "Home":
    st.title('Analysis on Airline Reviews')
    st.write('''Motivation:
The rationale behind scraping data from the airline reviews webpage is to construct a comprehensive dataset encompassing customer reviews, ratings, and pertinent details. This dataset serves multiple objectives, including:

Sentiment Analysis: Analyzing customer sentiments towards airlines through a comprehensive examination of their reviews and ratings.

Performance Evaluation: Assessing airlines' performance by analyzing customer feedback regarding aspects such as service quality, punctuality, and customer service.

Comparative Analysis: Conducting comparative assessments of airlines' performances by comparing their respective datasets.

Predictive Modeling: Developing machine learning models capable of predicting customer satisfaction or flight experiences based on review data.

Business Insights: Extracting actionable insights for airlines to improve their services, identify areas of enhancement, and enhance overall customer satisfaction.''')
elif nav_option == "Exploratory Data Analysis":
    st.title('Exploratory Data Analysis')
    st.subheader("Data")
    st.write("Reviews Fact")
    reviews_fact = get_dataframe("reviews_fact")
    st.dataframe(reviews_fact) 
    
    st.write("Airline Dimension")
    airline_dimension = get_dataframe("airline_dimension")
    st.dataframe(airline_dimension)

    # Univariate Analysis
    st.subheader("Univariate Analysis")

    # Compare Airline with Overall Rating
    plt.bar(airline_dimension["Airline"], airline_dimension["Overall Rating"])
    plt.xlabel('Airline')
    plt.ylabel('Overall Rating')
    plt.title('Overall Rating of Airlines')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    st.pyplot()

    #Count of each overall rating
    single_column_analysis(reviews_fact, "Overall Rating", "Overall Rating", 0, 0)

    # Trip verified
    single_column_analysis(reviews_fact, "Trip Verified", "Type of Verification", 1, 0)

    #Other reviews
    #Note sum of count are different because some rows do not have the review
    lst = ["Seat Comfort", "Cabin Staff Service", "Food & Beverages", "Ground Service", "Wifi & Connectivity", "Value For Money", "Inflight Entertainment"]
    for col in lst:
        single_column_analysis(reviews_fact, col, f"{col} Rating", 0, 0)

    #Recommended
    single_column_analysis(reviews_fact, "Recommended", "Status", 0, 0)

    #Origin
    single_column_analysis(reviews_fact, "Origin", "Origin Country", 1, 1)

    #Destination
    single_column_analysis(reviews_fact, "Destination", "Destination Country", 1, 1)

    # Cleaned number of characters
    review_len = reviews_fact['review_cleaned'].str.len()
    plt.figure(figsize=(16,8))
    plt.hist(review_len)
    plt.title('Number of Characters in Reviews')
    st.pyplot()

    # Cleaned number of words
    review_words = reviews_fact['review_cleaned'].str.split().map(lambda text: len(text))
    plt.figure(figsize=(16,8))
    plt.hist(review_words)
    plt.title('Number of Words in a review')
    st.pyplot()

    # Cleaned most common words overall
    corpus = create_corpus("review_cleaned", reviews_fact)
    counter = collections.Counter(corpus)
    most_common = counter.most_common()
    x, y = [], []
    for word, count in most_common[:30]:
        y.append(word)
        x.append(count)
    df = pd.DataFrame({'Count': x, 'Word': y})
    plt.figure(figsize=(16, 8))
    plt.title("Most common words in cleaned reviews")
    sns.barplot(x='Count', y='Word', data = df)
    st.pyplot()

    # Cleaned most common words respective airlines
    airline_lst = ["jetstar-airways","airasia","scoot","indigo-airlines","cebu-pacific","lion-air","zipair"]
    for airline in airline_lst:
        corpus = create_corpus("review_cleaned", reviews_fact[reviews_fact["Airline"] == airline])
        counter = collections.Counter(corpus)
        most_common = counter.most_common()
        x, y = [], []
        for word, count in most_common[:30]:
            y.append(word)
            x.append(count)
        df = pd.DataFrame({'Count': x, 'Word': y})
        plt.figure(figsize=(16, 8))
        plt.title(f"Most common words in cleaned {airline} reviews")
        sns.barplot(x='Count', y='Word', data = df)
        st.pyplot()

    # Cleaned wordcloud
    all_words = ' '.join(create_corpus("review_cleaned", reviews_fact))
    wordcloud = WordCloud(max_font_size=50, background_color='white').generate(all_words)
    plt.figure(figsize=(15,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title("Cleaned reviews wordcloud")
    st.pyplot()
    
    # Cleaned wordcloud for respective airlines
    airline_lst = ["jetstar-airways","airasia","scoot","indigo-airlines","cebu-pacific","lion-air","zipair"]
    for airline in airline_lst:
        airline_wordcloud(airline)


    st.subheader("Bivariate Analysis")
    
    # Box plot of overall rating of airlines
    airline_lst = ["jetstar-airways", "airasia", "scoot", "indigo-airlines", "cebu-pacific", "lion-air", "zipair"]
    data = {}
    for index, row in reviews_fact.iterrows():
        airline = row["Airline"]
        if airline in airline_lst:
            if airline not in data:
                data[airline] = []
            data[airline].append(row["Overall Rating"])
    # Create box plots for all airlines in the list
    plt.boxplot([data.get(airline, []) for airline in airline_lst], labels=airline_lst)
    # Adding labels and title
    plt.xlabel('Airlines')
    plt.ylabel('Overall Rating')
    plt.title('Box Plot of Overall Rating of Airlines')
    # Rotate the x-axis labels for better visibility
    plt.xticks(rotation=45)
    # Display the plot
    st.pyplot()

    # Average rating of each metric per airline
    grouped = reviews_fact[["Airline","Overall Rating","Seat Comfort","Cabin Staff Service","Food & Beverages","Ground Service","Wifi & Connectivity","Inflight Entertainment"]].groupby(['Airline']).mean().reset_index()
    metric_list = ["Seat Comfort","Cabin Staff Service","Food & Beverages","Ground Service","Wifi & Connectivity","Inflight Entertainment"]
    for metric in metric_list:
        lst = []
        for index, row in grouped.iterrows():
            lst.append((row["Airline"], row[metric]))
        lst.sort(key = lambda x: x[1])
        print(lst)
        plt.barh(list(map(lambda x: x[0], lst)), list(map(lambda x: x[1], lst)))
        plt.xlabel(metric)
        plt.ylabel('Airlines')
        plt.title(f"Average {metric} Per Airline")
        st.pyplot()  


elif nav_option == "ML":
    st.title("ML Results")
    #Data, accuracy and correlation matrix
    base_data = get_dataframe("baseML")
    st.subheader("Training Data")
    st.write("Reviews from years 2012-2016")
    st.dataframe(base_data)

    st.subheader("Baseline Model")
    st.write("Test Data: Reviews from years 2020-2024, with predicted sentiments from baseline model")
    baseline_data = get_dataframe("baseline_prediction")
    st.dataframe(baseline_data)

    pred_labels = baseline_data['sentiment_acc']
    true_labels = baseline_data['is_negative_sentiment']
    confusion_matrix_result = confusion_matrix(pred_labels, true_labels)
    st.write("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_result).plot(ax=ax)
    st.pyplot(fig)
    accuracy = accuracy_score(true_labels, pred_labels) * 100
    st.write(f"Accuracy score of baseline model on years 2020-2024 is {'{0:.2f}'.format(accuracy)}%")

    st.subheader("Finetune Model")
    st.write("Test Data: Reviews from years 2020-2024, with predicted sentiments from finetune model")
    finetune_data = get_dataframe("finetune_prediction")
    st.dataframe(finetune_data)

    pred_labels = finetune_data['sentiment_acc']
    true_labels = finetune_data['is_negative_sentiment']
    confusion_matrix_result = confusion_matrix(pred_labels, true_labels)
    st.write("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_result).plot(ax=ax)
    st.pyplot(fig)
    accuracy = accuracy_score(true_labels, pred_labels) * 100
    st.write(f"Accuracy score of finetune model on years 2020-2024 is {'{0:.2f}'.format(accuracy)}%")
else:
    st.write("Error")

st.write("Source: [airlinequality.com](http://www.airlinequality.com)")