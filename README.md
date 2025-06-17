 🚫 Deteksi Spam Bahasa Indonesia (Email/SMS)

Sistem berbasis web untuk mendeteksi pesan **SPAM atau HAM (bukan spam)** menggunakan Natural Language Processing (NLP) dan algoritma Machine Learning.

---

 🔍 Deskripsi Projek

Proyek ini merupakan aplikasi akhir dari mata kuliah Data Science yang bertujuan untuk:

- Mengembangkan sistem deteksi spam berbahasa Indonesia
- Menggunakan model machine learning (Naive Bayes)
- Mengolah pesan dari email & SMS
- Menyediakan antarmuka berbasis web menggunakan **FastAPI**
- Menampilkan visualisasi hasil klasifikasi secara real-time

---

 ⚙️ Teknologi yang Digunakan

| Komponen | Teknologi |
|----------|-----------|
| Bahasa Pemrograman | Python |
| Framework Web | FastAPI |
| Preprocessing | Sastrawi, NLTK |
| Model | Multinomial Naive Bayes |
| Vektorisasi | TF-IDF |
| Visualisasi | Matplotlib |
| UI Template | Flexy Bootstrap Admin Template |
| Deployment Lokal | Uvicorn |

Cara Menjalankan

1. Clone repositori ini:
```bash
git clone https://github.com/ariel3142/spam-detection.git
cd spam-detection/Web_Spam

2. Aktifkan virtual environment dan install dependensi:
pip install -r requirements.txt

3. Jalankan server FastAPI:
uvicorn main:app --reload
