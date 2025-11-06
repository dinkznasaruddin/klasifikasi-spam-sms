# Proyek: Klasifikasi Spam SMS (Naive Bayes vs. Logistic Regression)

## Ringkas
Pipeline end-to-end untuk mendeteksi spam SMS menggunakan dua model klasik: Multinomial Naive Bayes (MNB) dan Logistic Regression (LR), dengan prototipe Streamlit dan API Flask.

## Persiapan
1) Buat environment dan instal dependensi:
```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) Latih model dan hasilkan artefak:
```bash
python src_train.py
```
Artefak tersimpan di `artifacts/`:
- `champion_lr.joblib` atau `champion_nb.joblib` (tergantung hasil)
- `backup_nb.joblib` atau `backup_lr.joblib`
- `metrics.json`
- `confusion_matrix_test.png`

3) Jalankan Streamlit:
```bash
streamlit run app_streamlit.py
```

4) Jalankan Flask API:
```bash
python app_flask.py
# Uji dengan curl:
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Congratulations! You won a free prize!", "model":"lr", "threshold":0.5}'
```

## Catatan Replikasi
- Dataset otomatis diunduh dari UCI ML Repository (SMS Spam Collection) dan diekstrak ke `data/` saat menjalankan `src_train.py`.
- Seed ditetapkan ke 42 untuk konsistensi.
- Metrik terperinci tersedia pada `artifacts/metrics.json`.

## Struktur Repo
```
.
├── app_flask.py
├── app_streamlit.py
├── artifacts/            # dibuat saat training
├── data/                 # dataset UCI (diunduh otomatis)
├── requirements.txt
├── src_train.py
├── src_evaluate.py
└── paper_klasifikasi_spam_nb_vs_logreg.md
```

## Sumber & Lisensi Dataset
- Sumber: UCI Machine Learning Repository — SMS Spam Collection
  - URL: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
  - Unduhan: https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
- Sitasi:
  - Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering: New Collection and Results. Proceedings of the 11th ACM Symposium on Document Engineering, 259–262.
- Hak cipta dan ketentuan penggunaan mengikuti UCI ML Repository; mohon tinjau kebijakan UCI sebelum mendistribusikan ulang data.# klasifikasi-spam-sms
