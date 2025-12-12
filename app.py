from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import sqlite3
from datetime import datetime
import os

ARTIFACT_DIR = 'artifacts'
PIPELINE_PATH = os.path.join(ARTIFACT_DIR, 'pipeline.pkl')
DB_PATH = 'logs.db'

if not os.path.exists(PIPELINE_PATH):
    raise RuntimeError('Train model first (run train.py)')

pipeline = joblib.load(PIPELINE_PATH)
app = FastAPI(title='Customer Support Chatbot')

RESPONSES = {
    'refund': "Your refund request has been received. Refunds take 5–7 business days.",
    'billing': "There seems to be a billing issue. Please recheck payment details or contact billing support.",
    'account': "Try resetting your password. If the issue continues, send 'escalate'.",
    'technical': "We are investigating the technical issue. Try again after clearing cache.",
    'product': "Track your order in 'My Orders'. Visit product page for more info."
}

class Query(BaseModel):
    query: str

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT,
        category TEXT,
        confidence REAL,
        created_at TEXT
    )
''')
conn.commit()

@app.post('/predict')
def predict(q: Query):
    text = q.query.strip()
    if not text:
        raise HTTPException(status_code=400, detail='Empty query')

    probs = pipeline.predict_proba([text])[0]
    pred = pipeline.predict([text])[0]
    conf = float(np.max(probs))

    c.execute(
        'INSERT INTO logs (query, category, confidence, created_at) VALUES (?, ?, ?, ?)',
        (text, pred, conf, datetime.utcnow().isoformat())
    )
    conn.commit()

    if conf < 0.60:
        return {
            'category': None,
            'confidence': conf,
            'response': "The system is not confident. Connecting you to a human agent."
        }

    return {
        'category': pred,
        'confidence': conf,
        'response': RESPONSES.get(pred, "No response available.")
    }
