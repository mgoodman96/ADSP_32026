"""
score_headlines.py

Assignment 1: Michael Goodman

Purpose: use sklearn model developed by unknown data scientist to create an 
automated py file that can rate list of headlines with sentiment analysis

How to use: pass through the list of headlines you want to score along with 
its source, output file will be in directory with the source and current timestamp.
"""

import sys
import os

# Import libraries
import joblib
from sentence_transformers import SentenceTransformer
import pandas as pd


# static variables
today = pd.Timestamp.today().date()
year = today.year
month = today.month
day = today.day
clf = joblib.load("svm.joblib")
model = SentenceTransformer("All-MiniLM-L6-v2")


# load data
def load_data(file_path, source):
    """
    Loads the headlines from a file and returns the headlines and the source.

    Parameters:
    - file_path: str, path to the file containing headlines.
    - source: str, the source of the headlines (e.g., "NYT").

    Returns:
    - headlines: list of str, the headlines read from the file.
    - source: str, the source in lowercase.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        headlines = file.readlines()
    source = source.lower()
    return headlines, source


# apply model to headlines
def score_headlines(data):
    """
    Scores the headlines using the classifier and model.

    Parameters:
    - data: list of str, the headlines to be scored.
    - clf: the classifier model used to predict scores.
    - model: the sentence transformer model used to encode the headlines.

    Returns:
    - headline_scores: list of str, each headline with its predicted score.
    """
    headlines = data
    headline_scores = []
    for headline in headlines:
        headline_score = (clf.predict([model.encode(headline)]))[0]
        headline_scores.append(f"{headline}, {headline_score}")
    return headline_scores


# output
def save_headline_scores(headline_scores, source):
    """
    Saves the scored headlines to a text file with the given source and date.

    Parameters:
    - headline_scores: list of str, headlines with scores.
    - source: str, the source of the headlines.
    - year: int, the year to be included in the output file name.
    - month: int, the month to be included in the output file name.
    - day: int, the day to be included in the output file name.
    """
    with open(
        f"headline_scores_{source}_{year}_{month}_{day}.txt", "w", encoding="utf-8"
    ) as file:
        for headline_score in headline_scores:
            file.write(headline_score + "\n")


def main():
    """
    Main function to execute the script:
    1. Parses command-line arguments
    2. Loads data
    3. Scores the headlines
    4. Saves the results to a file
    """
    # add error handling. check if there are two arguments, check if file exists
    if len(sys.argv) != 3:
        print("Usage: python score_headlines.py <file_path> <source>")
        sys.exit(1)

    file_path = sys.argv[1]
    source = sys.argv[2]

    if not os.path.isfile(file_path):
        print(f"Error: The file {file_path} does not exist.")
        sys.exit(1)

    data, source = load_data(file_path, source)
    headline_scores = score_headlines(data)
    save_headline_scores(headline_scores, source)


if __name__ == "__main__":
    main()
