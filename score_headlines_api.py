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

# INFO:     Application startup complete.
# INFO:     75.86.52.153:61099 - "GET / HTTP/1.1" 404 Not Found
# INFO:     75.86.52.153:61099 - "GET /status HTTP/1.1" 200 OK
# INFO:     75.86.52.153:61105 - "GET /score_headlines?headlines=BAD HTTP/1.1" 422 Unprocessable Entity
# Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.49it/s]
# INFO:     75.86.52.153:61105 - "GET /score_headlines?headline=BAD HTTP/1.1" 200 OK
# INFO:     75.86.52.153:64112 - "GET /status HTTP/1.1" 200 OK
# INFO:     75.86.52.153:64112 - "GET /favicon.ico HTTP/1.1" 404 Not Found
# Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 25.22it/s]
# INFO:     75.86.52.153:61111 - "GET /score_headlines?headline=BAD HTTP/1.1" 200 OK
# INFO:     216.183.125.156:42989 - "GET /status HTTP/1.1" 200 OK
# INFO:     216.183.125.156:42989 - "GET /status HTTP/1.1" 200 OK
# INFO:     216.183.125.156:42989 - "GET /status HTTP/1.1" 200 OK
# INFO:     75.86.52.153:64146 - "GET /status HTTP/1.1" 200 OK
# ^CINFO:     Shutting down
# INFO:     Waiting for application shutdown.
# INFO:     Application shutdown complete.
# INFO:     Finished server process [423629]
# INFO:     Stopping reloader process [423625]