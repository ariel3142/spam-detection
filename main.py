from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk
import csv
import os

nltk.download('stopwords')

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

model = joblib.load('spam_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    cleaned_tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(cleaned_tokens)

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": ""})

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, message: str = Form(...)):
    clean = preprocess_text(message)
    vector = tfidf.transform([clean])
    prob = model.predict_proba(vector)[0]
    label = "SPAM" if model.predict(vector)[0] == 1 else "HAM (Bukan Spam)"
    confidence = prob[model.predict(vector)[0]] * 100
    result = f"{label} dengan confidence {confidence:.2f}%"

    # Simpan ke history.csv
    with open('history.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([message, result])

    return templates.TemplateResponse("index.html", {"request": request, "result": result})

@app.get("/history", response_class=HTMLResponse)
async def show_history(request: Request):
    history = []
    if os.path.exists('history.csv'):
        with open('history.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            history = list(reader)
    return templates.TemplateResponse("history.html", {"request": request, "history": history})
