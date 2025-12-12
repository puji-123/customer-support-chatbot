Machine Learning + NLP + FastAPI + Python

An intelligent customer support chatbot that automatically classifies user queries (billing issues, refund requests, login problems, etc.) and provides accurate responses using NLP + ML.
This project is built end-to-end with Python, Scikit-learn, FastAPI, and TF-IDF text processing.

⭐ Features
✔ 1. NLP Preprocessing

Tokenization

Stopword removal

Lemmatization (optional)

TF-IDF vectorization

✔ 2. ML-based Query Classification

Trained on labelled customer support data to detect:

Billing Issues

Technical Issues

Login Problems

Refund Requests

Product/Order Information

Models used:

Logistic Regression

Support Vector Machine

Naive Bayes (optional)

✔ 3. Automated Response Generation

Based on predicted class, the bot returns a helpful predefined response.

✔ 4. Confidence Threshold

If prediction confidence < 60%, system auto-escalates to human support.

✔ 5. REST API with FastAPI

Predict endpoint:

POST /predict
{
  "query": "I can't login to my account"
}

✔ 6. Production-Ready Pipeline

The model is stored as:

artifacts/pipeline.pkl

✔ 7. Logging

All predictions stored in SQLite (logs.db) for analytics.

🏗️ Project Architecture
                +---------------------+
User Query ---> |  FastAPI Endpoint  |
                +---------+-----------+
                          |
                          v
                +---------------------+
                |  ML Pipeline (pkl) |
                |  TF-IDF + Model    |
                +---------+-----------+
                          |
                          v
                +---------------------+
                | Response Generator |
                +---------------------+

📂 Folder Structure
project/
│
├── app.py                 # FastAPI app for model inference
├── train.py               # Training script (vectorizer + classifier)
├── requirements.txt       # Dependencies
├── pipeline.pkl           # Saved ML pipeline
├── logs.db                # Logs for analytics
└── README.md              # Project documentation

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/puji-123/customer-support-chatbot.git
cd customer-support-chatbot

2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

🧠 Train the Model

You can retrain the model anytime using:

python train.py


This regenerates pipeline.pkl.

🚀 Run FastAPI Server
uvicorn app:app --reload


Server starts at:

👉 http://127.0.0.1:8000

Swagger docs available at:

👉 http://127.0.0.1:8000/docs

📌 Example Prediction
Request:
POST /predict
{
  "query": "My payment failed yesterday"
}

Response:
{
  "category": "billing",
  "response": "There seems to be a billing issue. Please recheck payment details."
}

📊 Dataset Used

A real-world-style dataset of customer support queries labelled into categories:

Query Example	Label
“My payment is not going through”	billing
“I want a refund for this order”	refund
“App is crashing on start”	technical
“I forgot my password”	account
“How do I track my order?”	product

Dataset size: 500 rows (can be expanded)

💡 Future Improvements

Add BERT / Transformer-based models

Multilingual support

Response ranking using similarity search

Integrate with WhatsApp / Telegram

Add web dashboard for analytics

🤝 Why This Project Is Great for Interviews

Recruiters love this project because:

✔ Shows real-world ML/NLP
✔ Demonstrates API creation
✔ Includes ML pipeline architecture
✔ Easy to explain in 2–3 minutes
✔ Proves practical Python + FastAPI skills

This is a portfolio-worthy, production-style project.

🧑‍💻 Author

Pujitha Reddy (puji-123)
Feel free to star ⭐ this repo if you found it helpful.
