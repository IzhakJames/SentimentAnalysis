#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import traceback
import math


# In[ ]:


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


# In[ ]:


# Initialize an empty DataFrame to store the comments data
comments_data = pd.DataFrame(columns=['Airline','Review ID','Date Published', 'Overall Rating', 'Passenger Country', 'Trip Verified', 'Review Title','Review', 
                                       'Aircraft', 'Type Of Traveller', 'Seat Type', 'Origin', 'Destination', 'Layover', 'Date Flown', 
                                       'Seat Comfort', 'Cabin Staff Service', 'Food & Beverages', 'Ground Service', 
                                       'Value For Money', 'Recommended'])
comments_data_list = [] 


# In[ ]:


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


# In[ ]:


lst_names = ["scoot","zipair","airasia","cebu-pacific","lion-air","indigo-airlines","jetstar-airways"]
for name in lst_names:
    get_airline_data(name)
    
# Convert the list of dictionaries into a DataFrame
comments_data = pd.DataFrame(comments_data_list)
comments_data.to_csv('all_reviews.csv', encoding='utf-8', index=False)


# In[ ]:




