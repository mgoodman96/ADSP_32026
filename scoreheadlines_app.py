"""
Streamlit app developed for ADSP 32026 - Score Headlines Project

This app allows users to input a headline and receive a score based on the sentiment of the headline. 
The sentiment is determined by a pre-trained SVM model that uses a pre-trained sentence transformer model to encode the headline and return a score.

The app is built using Streamlit and the FastAPI API.

"""

#import libraries
import streamlit as st
import requests
import json
import logging

# Set up the API URL
API_URL = "http://localhost:8080"

def main(API_URL):
    # Set up the Streamlit app
    st.title("Score Headlines App")
    st.write("Author: Michael Goodman")
    st.write("This app allows users to input a headline and receive a score based on the sentiment of the headline. The sentiment is determined by a pre-trained SVM model that uses a pre-trained sentence transformer model to encode the headline and return a score.")
    st.write("The app is built using Streamlit and the FastAPI API.")
    st.write("To use the app, simply enter a headline in the input field above and click 'Enter'. The app will then return a score based on the sentiment of the headline.")
    #

    # Set up the input field for the headline
    headline = st.text_input("Enter a headline:")
    if headline:
        # Make a request to the API to get the score
        response = requests.get(f"{API_URL}/score_headlines?headline={headline}")
        if response.status_code == 200:
            # Parse the response and display the score
            score = json.loads(response.text)["score"]
            st.write(f"The headline sentiment is: '{score}")
        else:
            st.write("Error getting score. Please try again.")
            st.write(logging.error(response.text))

if __name__ == "__main__":
    main(API_URL)
    

#deploy
#streamlit run scoreheadlines_app.py "http://localhost:8080" --server.port 9086
