from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
import sys
import logging
import joblib
from sentence_transformers import SentenceTransformer
import pandas as pd

'''Set up logging'''
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

'''Load the model and classifier'''
logger.info("Loading model and classifier")
clf = joblib.load("svm.joblib")
model = SentenceTransformer("All-MiniLM-L6-v2")

'''Run App'''

app = FastAPI()
class Headline(BaseModel):
    headline: str

@app.get('/status')
def status():
    return {'status': 'ok'}

@app.get('/score_headlines')
def score_headlines(headline: str):
    headline_score = clf.predict([model.encode(headline)])[0]
    return {'score': headline_score}

#py -m uvicorn score_headlines_api:app --host localhost --port 8080 --reload