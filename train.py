# train.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os


DATA_PATH = 'data/customer_support.csv'
ARTIFACT_DIR = 'artifacts'
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# Load dataset
df = pd.read_csv(DATA_PATH)
X = df['query'].astype(str)
y = df['label'].astype(str)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


pipeline = Pipeline([
('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=1)),
('clf', LogisticRegression(max_iter=1000))
])


pipeline.fit(X_train, y_train)


# Evaluate
preds = pipeline.predict(X_test)
print(classification_report(y_test, preds))


# Save
joblib.dump(pipeline, os.path.join(ARTIFACT_DIR, 'pipeline.pkl'))
print('Saved pipeline to artifacts/pipeline.pkl')