from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import re
import csv
import os
import matplotlib.pyplot as plt
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk
import pandas as pd

nltk.download('stopwords')

app = FastAPI()

# Direktori dan template
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model dan vectorizer
model = joblib.load('spam_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    cleaned_tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(cleaned_tokens)

def generate_pie_chart():
    if not os.path.exists('history.csv'):
        return None
    df = pd.read_csv('history.csv', names=["text", "result"])
    if df.empty:
        return None
    latest_result = df.iloc[-1]["result"]
    match = re.search(r'confidence ([\d.]+)%', latest_result)
    confidence = float(match.group(1)) if match else 0.0
    is_spam = latest_result.startswith("SPAM")

    if is_spam:
        spam_pct = confidence
        ham_pct = 100 - confidence
    else:
        ham_pct = confidence
        spam_pct = 100 - confidence

    fig, ax = plt.subplots()
    ax.pie(
        [spam_pct, ham_pct],
        labels=["SPAM", "HAM"],
        autopct='%1.1f%%',
        colors=["#ff6666", "#66b3ff"]
    )
    plt.title("Distribusi Hasil Prediksi Terakhir")
    chart_path = "static/pie_chart.png"
    plt.savefig(chart_path)
    plt.close()
    return chart_path

def generate_bar_chart():
    if not os.path.exists('history.csv'):
        return None
    df = pd.read_csv('history.csv', names=["text", "result"])

    label_counts = Counter()
    for r in df['result']:
        if r.startswith("SPAM"):
            label_counts["SPAM"] += 1
        else:
            label_counts["HAM"] += 1

    fig, ax = plt.subplots()
    ax.bar(label_counts.keys(), label_counts.values(), color=["#ff6666", "#66b3ff"])
    ax.set_title("Jumlah Prediksi per Label")
    chart_path = "static/bar_chart.png"
    plt.savefig(chart_path)
    plt.close()
    return chart_path

def generate_confidence_chart():
    if not os.path.exists('history.csv'):
        return None
    df = pd.read_csv('history.csv', names=["text", "result"])
    if df.empty:
        return None
    latest_result = df.iloc[-1]["result"]
    match = re.search(r'confidence ([\d.]+)%', latest_result)
    confidence = float(match.group(1)) if match else 0.0
    fig, ax = plt.subplots()
    ax.plot([0, 1], [confidence, confidence], color="#ffc107", linewidth=4)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 1)
    ax.set_title("Confidence Prediksi Terakhir")
    chart_path = "static/conf_chart.png"
    plt.savefig(chart_path)
    plt.close()
    return chart_path

def generate_word_freq_chart():
    if not os.path.exists('history.csv'):
        return None
    df = pd.read_csv('history.csv', names=["text", "result"])
    all_text = " ".join(df["text"])
    text = preprocess_text(all_text)
    tokens = text.split()
    word_freq = Counter(tokens).most_common(10)
    if not word_freq:
        return None
    words, counts = zip(*word_freq)
    fig, ax = plt.subplots()
    ax.barh(words[::-1], counts[::-1], color="#17a2b8")
    ax.set_title("10 Kata Paling Sering")
    chart_path = "static/word_freq.png"
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    return chart_path

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, message: str = Form(...)):
    clean = preprocess_text(message)
    vector = tfidf.transform([clean])
    prob = model.predict_proba(vector)[0]
    label_index = model.predict(vector)[0]
    label = "SPAM" if label_index == 1 else "HAM (Bukan Spam)"
    confidence = prob[label_index] * 100
    result = f"{label} dengan confidence {confidence:.2f}%"

    with open('history.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([message, result])

    return RedirectResponse(url="/result", status_code=303)

@app.get("/result", response_class=HTMLResponse)
async def show_result(request: Request):
    result = ""
    if os.path.exists("history.csv"):
        with open("history.csv", "r", encoding="utf-8") as f:
            lines = list(csv.reader(f))
            if lines:
                last = lines[-1]
                result = last[1] if len(last) > 1 else ""

    return templates.TemplateResponse("result.html", {
        "request": request,
        "result": result,
        "chart_path": generate_pie_chart(),
        "bar_chart": generate_bar_chart(),
        "conf_chart": generate_confidence_chart(),
        "word_chart": generate_word_freq_chart()
    })

@app.get("/history", response_class=HTMLResponse)
async def show_history(request: Request):
    history = []
    if os.path.exists('history.csv'):
        with open('history.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            history = list(reader)
    return templates.TemplateResponse("history.html", {"request": request, "history": history})

@app.get("/clear-history")
async def clear_history():
    if os.path.exists("history.csv"):
        os.remove("history.csv")
    return RedirectResponse(url="/history", status_code=303)
