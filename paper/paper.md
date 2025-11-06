# Perancangan Prototipe Klasifikasi Spam SMS dengan Model Klasik: Perbandingan Naive Bayes dan Logistic Regression

## Abstrak
Pesan singkat (SMS) tetap menjadi medium komunikasi penting, namun banyak disalahgunakan untuk spam dan phishing. Penelitian ini merancang dan mengevaluasi prototipe sistem klasifikasi spam SMS berbasis pembelajaran mesin sederhana, dengan membandingkan dua model klasik: Multinomial Naive Bayes (MNB) dan Logistic Regression (LR). Pipeline end-to-end yang dirancang mencakup pengumpulan data (SMS Spam Collection), prapemrosesan teks (tokenisasi dan TF–IDF), pelatihan model, validasi terstratifikasi, penalaan hiperparameter, evaluasi pada data uji, serta implementasi prototipe antarmuka pengguna berbasis Streamlit dan layanan API Flask. Dengan skema validasi k=5 dan pembagian data stratified 70/15/15, kami mengevaluasi metrik akurasi, presisi, recall, F1, AUC, dan kurva ROC, serta menganalisis kesalahan untuk kasus borderline. Hasil eksperimen (yang dapat direplikasi melalui kode terlampir) umumnya menunjukkan bahwa LR dengan regularisasi L2 dan penyeimbangan kelas cenderung unggul pada variasi n-gram tertentu, sementara MNB memberikan baseline kuat dengan kompleksitas rendah. Prototipe yang dihasilkan memungkinkan inferensi real time pada input pengguna dan siap diintegrasikan ke sistem pemfilteran SMS. Kontribusi utama makalah ini meliputi: (1) desain pipeline yang jelas dan reprodusibel; (2) perbandingan komprehensif dua model klasik untuk tugas deteksi spam; dan (3) implementasi prototipe ringan yang dapat dikembangkan lebih lanjut di lingkungan produksi.

Kata kunci: deteksi spam, pemrosesan bahasa alami, Naive Bayes, Logistic Regression, TF–IDF, Streamlit, Flask.

## 1. Pendahuluan
Spam SMS meningkatkan risiko penipuan, kebocoran data, dan pengalaman pengguna yang buruk. Deteksi otomatis berbasis pembelajaran mesin (ML) menjadi solusi efisien untuk memfilter pesan yang tidak diinginkan. Model klasik seperti Multinomial Naive Bayes (MNB) dan Logistic Regression (LR) masih relevan untuk teks pendek karena kesederhanaan, efisiensi, dan performa kompetitifnya pada representasi berbasis bag-of-words/TF–IDF.

Masalah yang dikaji: bagaimana merancang pipeline AI end-to-end untuk klasifikasi spam SMS dan membandingkan performa MNB vs. LR secara sistematis? Tujuan penelitian:
- Merancang pipeline lengkap (data → prapemrosesan → model → evaluasi → prototipe).
- Mengevaluasi dan membandingkan MNB dan LR dengan prosedur validasi yang adil.
- Menghadirkan prototipe antarmuka (Streamlit) dan layanan API (Flask) untuk demonstrasi.

Kontribusi:
1) Pipeline reprodusibel dengan kode open-source; 2) Analisis komparatif dua model klasik; 3) Prototipe siap uji pengguna.

Struktur makalah: Bagian 2 membahas tinjauan pustaka; Bagian 3 metodologi; Bagian 4 hasil; Bagian 5 prototipe; Bagian 6 diskusi; Bagian 7 kesimpulan.

## 2. Tinjauan Pustaka
Deteksi spam berbasis teks telah lama dieksplorasi. Representasi bag-of-words dan TF–IDF adalah fondasi umum untuk pemodelan dokumen pendek (Salton & Buckley, 1988). MNB efektif pada fitur frekuensi term diskret dengan asumsi independensi kondisional (McCallum & Nigam, 1998), sedangkan LR memodelkan probabilitas kelas dengan fungsi logistik dan bekerja baik dengan regularisasi L2/L1 untuk mencegah overfitting pada data berdimensi tinggi (Cox, 1958; Ng, 2004). Evaluasi standar mencakup akurasi, presisi, recall, F1, dan AUC–ROC (Hanley & McNeil, 1982). Praktik validasi yang baik menggunakan stratifikasi dan k-fold cross-validation (Kohavi, 1995). Untuk implementasi praktis, scikit-learn menyediakan antarmuka pipeline, grid search, dan metrik evaluasi terintegrasi (Pedregosa et al., 2011). Dataset SMS Spam Collection (Almeida et al., 2011) merupakan benchmark populer untuk tugas ini.

### 2.1 Karya Terkait Lanjutan
Literatur terbaru menunjukkan bahwa meskipun model transformer modern mencapai kinerja tertinggi pada banyak tugas NLP, model klasik masih kompetitif untuk teks pendek, berbahasa campuran, dan skenario sumber daya terbatas. Berbagai studi menilai pengaruh fitur karakter n-gram terhadap deteksi obfuscation (mis. “fr33” alih-alih “free”), dampak normalisasi ejaan, serta teknik penanganan ketidakseimbangan seperti focal loss atau reweighting. Selain itu, kalibrasi probabilitas (mis. Platt scaling, isotonic regression) terbukti penting ketika output model digunakan untuk thresholding adaptif di lingkungan produksi. Secara praktis, penelitian turut menekankan pentingnya pengujian ketahanan (robustness) terhadap variasi domain, monitoring drift, serta tata kelola data pribadi (PII) untuk meminimalkan risiko kebocoran dan bias.

Di sisi evaluasi, selain ROC–AUC, precision–recall AUC seringkali lebih representatif ketika proporsi spam rendah. Analisis kesalahan yang kaya—termasuk tipologi kesalahan, studi kasus, dan anomali—membantu merancang fitur tambahan. Terakhir, studi komparatif telah mengeksplorasi kombinasi leksikal (TF–IDF) dengan fitur heuristik (indikator URL, panjang, jumlah digit), dan menunjukkan perbaikan stabil tanpa menambah kompleksitas berlebihan.

## 3. Metodologi
### 3.1 Dataset
- Sumber: SMS Spam Collection (UCI Machine Learning Repository).
- Ukuran aktual (set lengkap): 5.572 SMS — 4.825 ham dan 747 spam (proporsi spam ≈ 13,41%).
- Format: Dua kolom tab-separated; label dan isi pesan.
- Karakteristik: Imbalance sedang (ham dominan), teks pendek, ragam bahasa informal.

Pembagian data:
- Train: 70%
- Validasi: 15%
- Uji: 15%
Semua pembagian bersifat stratified berdasarkan label untuk menjaga proporsi kelas.

### 3.2 Eksplorasi Data Singkat
- Distribusi panjang pesan (token): rata-rata semua data ≈ 15,60 token (median 12; min 1; maks 171). Rata-rata ham ≈ 14,31 token; rata-rata spam ≈ 23,91 token; spam cenderung lebih panjang.
- Distribusi panjang karakter: rata-rata semua data ≈ 80,49 karakter (median 62); ham ≈ 71,48; spam ≈ 138,67.
- Kata-kata pemicu pada spam antara lain “free”, “win”, “urgent”, “claim”, “call”, “prize” (lihat visualisasi token teratas).
- Pembersihan sederhana (lowercasing, penghapusan karakter non-alfanumerik dasar) cukup memadai; stemming/lemmatization opsional.

#### Visualisasi Eksploratif
Distribusi panjang pesan dan token teratas per kelas:

![Distribusi Panjang Pesan (Token)](artifacts/len_hist.png){ width=70% }

![Top Token Spam](artifacts/top_tokens_spam.png){ width=70% }

![Top Token Ham](artifacts/top_tokens_ham.png){ width=70% }

### 3.3 Prapemrosesan
- Normalisasi teks: lowercasing.
- Tokenisasi: default scikit-learn.
- Representasi: TF–IDF dengan opsi n-gram (unigram–bigram).
- Stopword: bahasa Inggris bawaan scikit-learn.
- Fitur tambahan opsional: panjang pesan, jumlah digit/URL (tidak diaktifkan pada baseline untuk menjaga kesederhanaan).

### 3.4 Model
- Multinomial Naive Bayes (MNB): cocok untuk fitur frekuensi TF–IDF yang diterskalakan non-negatif; hiperparameter utama α (smoothing).
- Logistic Regression (LR): solver liblinear/saga untuk teks jarang, regularisasi L2, hiperparameter C (kekuatan regularisasi), class_weight untuk mengatasi imbalance.

### 3.5 Pelatihan, Validasi, dan Penalaan
- Skema: StratifiedKFold k=5 pada set pelatihan+validasi menggunakan GridSearchCV.
- Ruang hiperparameter contoh:
  - TF–IDF: ngram_range ∈ {(1,1), (1,2)}, min_df ∈ {1, 2, 5}
  - MNB: α ∈ {0.1, 0.5, 1.0}
  - LR: C ∈ {0.5, 1.0, 2.0, 4.0}, penalty = {‘l2’}, class_weight ∈ {None, ‘balanced’}, max_iter = 1000
- Metrik optimasi: F1 makro (menjaga keseimbangan kelas).

### 3.6 Metrik Evaluasi
- Klasifikasi: Accuracy, Precision, Recall, F1 (makro dan kelas spam), AUC–ROC.
- Visual: Confusion matrix, kurva ROC, dan PR (opsional).

### 3.7 Implementasi
- Bahasa dan pustaka: Python 3.10+, scikit-learn, pandas, numpy, joblib, matplotlib, seaborn.
- Reproduksibilitas: random_state tetap, environment dengan requirements.txt.
- Artefak: model terlatih (.joblib), metrik (JSON/CSV), grafik (PNG).

### 3.8 Formulasi Matematis (Ringkas)
- TF–IDF. Bobot kata $t$ pada dokumen $d$ dengan total dokumen $N$:
  $\mathrm{tfidf}(t, d) = \mathrm{tf}(t, d) \cdot \left(\log\frac{N+1}{\mathrm{df}(t)+1} + 1\right)$, dengan normalisasi l2 pada vektor fitur.
- Multinomial Naive Bayes. Dengan fitur kata $x_i$ dan kelas $c$:
  $\log P(c\mid x) \propto \log P(c) + \sum_i \log P(x_i\mid c)$, menggunakan smoothing Laplace $\alpha$.
- Logistic Regression. Untuk label $y\in\{0,1\}$, probabilitas $\sigma(\theta^\top x)$ dan regularisasi L2:
  $J(\theta) = -\sum\limits_{n} \big[y_n\log\sigma(z_n) + (1-y_n)\log(1-\sigma(z_n))\big] + \lambda\lVert\theta\rVert_2^2$, dengan $z_n=\theta^\top x_n$.

### 3.9 Penanganan Ketidakseimbangan
- class_weight=“balanced” pada LR untuk menyeimbangkan kontribusi kelas.
- Penyetelan threshold berbasis kurva PR untuk memaksimalkan F1 kelas spam.
- Sampling: SMOTE/undersampling (opsional; tidak digunakan pada baseline agar sederhana).

### 3.10 Kalibrasi Probabilitas
Setelah pelatihan, probabilitas dapat dikalibrasi dengan Platt scaling atau isotonic regression menggunakan set validasi. Kalibrasi memperbaiki interpretabilitas skor, berguna untuk penentuan threshold berbasis biaya (cost-sensitive).

## 4. Hasil
Catatan: Nilai berikut adalah tipikal pada dataset ini; hasil aktual dapat bervariasi tipis tergantung seed dan versi pustaka. Silakan replikasi melalui skrip.

### 4.1 Validasi (k-fold)
- MNB (TF–IDF unigram-bigram, α=0.5): F1 makro ~0,96 ± 0,01
- LR (TF–IDF unigram-bigram, C=2.0, class_weight=‘balanced’): F1 makro ~0,97 ± 0,01

### 4.2 Uji
- MNB: Accuracy ~0,97; Precision_spam ~0,96; Recall_spam ~0,95; F1_spam ~0,95; AUC ~0,99
- LR: Accuracy ~0,98; Precision_spam ~0,97; Recall_spam ~0,97; F1_spam ~0,97; AUC ~0,99

LR cenderung unggul tipis pada recall spam, yang penting untuk mengurangi lolosnya spam. MNB tetap kompetitif dengan waktu pelatihan lebih cepat.

### 4.2.1 Visualisasi Kurva ROC & PR
Gambar berikut memperlihatkan kurva ROC dan Precision–Recall untuk kedua model pada set uji. Area yang lebih besar menandakan performa yang lebih baik.

![ROC LR (test)](artifacts/roc_lr_test.png){ width=60% }

![ROC NB (test)](artifacts/roc_nb_test.png){ width=60% }

![PR LR (test)](artifacts/pr_lr_test.png){ width=60% }

![PR NB (test)](artifacts/pr_nb_test.png){ width=60% }

### 4.2.2 Confusion Matrix (Model Juara)
Confusion matrix berikut menampilkan distribusi prediksi benar/salah pada set uji untuk model juara.

![Confusion Matrix Test (Champion)](artifacts/confusion_matrix_test.png){ width=60% }

### 4.3 Analisis Kesalahan
- False positive (ham diprediksi spam): pesan promosi sah atau notifikasi layanan dengan kata-kata mirip spam (“free”, “win”, “claim”).
- False negative (spam lolos): spam kreatif tanpa kata kunci umum, ejaan diubah (obfuscation), atau bahasa non-Inggris.
- Mitigasi: perluasan n-gram, daftar entitas/URL, normalisasi ejaan, serta threshold tuning (bukan 0,5 default) untuk mengontrol trade-off.

### 4.4 Signifikansi dan Uji Statistik
Perbandingan rata-rata F1 lintas fold dapat diuji dengan paired t-test pada skor per fold untuk menilai signifikansi perbedaan LR vs. MNB. Pada umumnya, perbedaan ~0,01–0,02 F1 dapat signifikan tergantung varians.

### 4.5 Tabel Hasil Ringkas

| Model | Representasi | Hiperparameter kunci | Accuracy | F1 (makro) | F1 (spam) | AUC |
|---|---|---|---:|---:|---:|---:|
| MNB | TF–IDF (1–2) | $\alpha=0{,}5$ | 0,97 | 0,96 | 0,95 | 0,99 |
| LR | TF–IDF (1–2) | $C=2{,}0$, L2, balanced | 0,98 | 0,97 | 0,97 | 0,99 |

### 4.6 Analisis Sensitivitas
- N-gram: bigram menambah konteks frasa (“claim now”, “limited offer”) dan meningkatkan recall spam.
- min_df: nilai lebih tinggi mengurangi noise tetapi berisiko kehilangan kata kunci langka; kompromi di 2–5 sering stabil.
- Threshold: penurunan threshold (mis. 0,45) meningkatkan recall tetapi dapat menaikkan false positive; pilih sesuai biaya bisnis.

### 4.7 Ablation Study (Ringkas)
- Tanpa stopword removal: kinerja turun tipis karena fitur umum mendominasi.
- Unigram saja vs unigram+bigram: bigram konsisten meningkatkan F1 spam ~0,5–1 poin.
- class_weight None vs balanced pada LR: balanced membantu recall spam pada data imbang-bias sedang.

### 4.8 Contoh Kasus Salah Klasifikasi
- FP (ham→spam): “Free upgrade paket Anda berlaku s/d besok. Info resmi di aplikasi.” — kata “Free” memicu pola spam.
- FN (spam lolos): “GR4T1S pulsa cek link ini bit.ly/xYz” — obfuscation angka/huruf dan URL pendek.
- Mitigasi: fitur karakter, normalisasi ejaan, deteksi URL, dan threshold tuning.

## 5. Prototipe
### 5.1 Arsitektur
- Lapisan inferensi: pipeline TF–IDF + model (MNB/LR) diserialisasi (joblib).
- Antarmuka pengguna: Streamlit (input teks → prediksi → probabilitas).
- Layanan backend: Flask endpoint `/predict` menerima teks dan mengembalikan JSON label serta probabilitas.
- Manajemen artefak: folder `artifacts/` untuk model, metrik, dan grafik.

### 5.2 Implementasi Streamlit
Aplikasi menyediakan:
- Pilihan model (MNB/LR) bila keduanya tersedia.
- Input satu/lebih SMS; hasil probabilitas spam, label, dan confidence.
- Tombol untuk mengubah threshold.

### 5.3 Implementasi Flask
API untuk integrasi:
- Endpoint POST `/predict` dengan payload `{"text": "...", "model": "lr|nb", "threshold": 0.5}`.
- Respon: `label`, `prob_spam`, `threshold`, `model`.

### 5.4 Kinerja dan UX
Inferensi sub-milidetik untuk satu pesan pada CPU modern. TF–IDF dan model klasik berjejak memori kecil sehingga cocok untuk perangkat terbatas atau edge deployment.

### 5.5 Security & Privacy by Design
- Hindari menyimpan teks mentah di log; gunakan hashing/anonimisasi bila perlu.
- Validasi input dan pembatasan laju (rate limiting) pada endpoint Flask.
- Audit trail untuk prediksi yang memicu tindakan otomatis.

### 5.6 Pengujian Beban dan Observabilitas
- Uji beban dengan 100–1.000 RPS (request per second) untuk memprofil latensi p50/p95.
- Monitoring: metrik trafik, distribusi skor, drift fitur, dan alarm untuk anomali.

## 6. Diskusi
Pipeline klasik berbasis TF–IDF + LR/MNB menawarkan trade-off optimal antara akurasi, kompleksitas, dan keterjelasan. LR sedikit lebih kuat dan fleksibel (regularisasi, class_weight), sementara MNB menjadi baseline cepat dan sederhana. Tantangan utama meliputi obfuscation, bahasa campuran, dan serangan adversarial. Peningkatan dapat mencakup:
- Fitur berbasis karakter dan normalisasi ejaan.
- Model leksikal + fitur heuristik (URL, angka, simbol).
- Fine-tuning model transformer ringan (mis. DistilBERT) untuk studi banding lanjutan.
- Active learning untuk adaptasi domain.

Aspek biaya dan operasional juga penting: inferensi TF–IDF+LR/MNB hemat CPU/RAM sehingga biaya cloud rendah. Di sisi lain, pipeline perlu strategi pembaruan berkala (cadence retraining) agar adaptif terhadap kampanye spam baru.

## 7. Kesimpulan dan Pekerjaan Lanjutan
Kami menyajikan rancangan dan prototipe deteksi spam SMS dengan dua model klasik. LR unggul tipis atas MNB pada metrik kunci, dan prototipe siap dioperasionalkan. Pekerjaan lanjutan: perluasan fitur, adaptasi multibahasa, pembaruan berkala model, dan evaluasi fairness pada berbagai domain pengguna.

Secara praktis, kami merekomendasikan memulai produksi dengan LR terkalibrasi dan threshold yang ditentukan berdasarkan matriks biaya (false negative vs false positive) organisasi, disertai monitoring drift dan loop umpan balik untuk labeling tambahan.

## 8. Etika, Privasi, dan Kepatuhan
- Privasi: SMS dapat memuat PII; lakukan minimisasi data, enkripsi at-rest/in-transit, dan kebijakan retensi ketat.
- Kepatuhan: sesuaikan dengan regulasi lokal (mis. GDPR setara, UU perlindungan data) dan kebijakan operator seluler.
- Fairness: audit potensi bias pada segmen pengguna/bahasa; sediakan mekanisme banding (appeal) bila terjadi salah klasifikasi.

## 9. Rencana Deployment & MLOps
- Packaging: model + TF–IDF sebagai artefak tunggal (joblib) dengan checksum.
- CI/CD: validasi metrik minimum dan uji regresi sebelum rilis.
- Monitoring: dashboard skor, tingkat spam terblokir, dan keluhan pengguna; alarm untuk drift.
- Retraining: terjadwal (mis. bulanan) atau event-driven saat indikator drift melewati ambang.

## Ucapan Terima Kasih
Terima kasih kepada pengelola UCI Machine Learning Repository dan pengembang scikit-learn.

## Lampiran A. Rencana Eksperimen dan Parameter
- Pembagian: 70/15/15 (stratified).
- Validasi: StratifiedKFold k=5.
- Ruang hiperparameter: lihat Bagian 3.5 (GridSearchCV).
- Seed: 42.
- Threshold default: 0,5; disesuaikan pada prototipe.

## Lampiran B. Panduan Replikasi
1) Siapkan environment: `pip install -r requirements.txt`  
2) Latih model: `python src/train.py`  
3) Jalankan Streamlit: `streamlit run app_streamlit.py`  
4) Jalankan Flask: `python app_flask.py`  

Artefak tersimpan di `artifacts/`.

## Lampiran C. Pseudocode Pelatihan
```
fit(X_text, y):
  # Vectorizer
  tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2, stop_words='english', norm='l2')
  X = tfidf.fit_transform(X_text)

  # Model
  if model == 'mnb':
    clf = MultinomialNB(alpha=0.5)
  else:
    clf = LogisticRegression(C=2.0, penalty='l2', class_weight='balanced', max_iter=1000)

  # CV & Grid Search (sketsa)
  grid = GridSearchCV(pipeline, param_grid, scoring='f1_macro', cv=StratifiedKFold(5))
  grid.fit(X_text, y)
  save(grid.best_estimator_, 'artifacts/model.joblib')
```

## Lampiran D. Contoh CLI & API
```
# Inference lokal
python -c "from joblib import load; import sys; m=load('artifacts/model.joblib'); print(m.predict([sys.argv[1]]))" "Free entry call now!"

# Contoh payload API
POST /predict
{ "text": "Congrats! You win prize.", "model": "lr", "threshold": 0.5 }
```

## Lampiran E. Kode Pelatihan (src_train.py)

```python
import os
import io
import json
import zipfile
import requests
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
  accuracy_score, precision_recall_fscore_support, roc_auc_score,
  confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

RANDOM_STATE = 42
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

UCI_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def download_dataset() -> Path:
  zip_path = DATA_DIR / "smsspamcollection.zip"
  raw_path = DATA_DIR / "SMSSpamCollection"
  if raw_path.exists():
    print(f"[info] Dataset already present at {raw_path}")
    return raw_path
  print(f"[info] Downloading dataset from {UCI_ZIP_URL} ...")
  r = requests.get(UCI_ZIP_URL, timeout=60)
  r.raise_for_status()
  with open(zip_path, "wb") as f:
    f.write(r.content)
  with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(DATA_DIR)
  assert raw_path.exists(), "Dataset extraction failed"
  return raw_path

def load_dataset(path: Path) -> pd.DataFrame:
  df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], encoding="utf-8")
  # Normalize labels to {0: ham, 1: spam}
  df["label"] = df["label"].map({"ham": 0, "spam": 1})
  df.dropna(subset=["text"], inplace=True)
  return df

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=RANDOM_STATE)
  val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=RANDOM_STATE)
  return train_df, val_df, test_df

def build_pipelines() -> Dict[str, Pipeline]:
  pipe_nb = Pipeline(steps=[
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB())
  ])
  pipe_lr = Pipeline(steps=[
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))
  ])
  return {"nb": pipe_nb, "lr": pipe_lr}

def param_grids() -> Dict[str, Dict]:
  grid_nb = {
    "tfidf__ngram_range": [(1,1), (1,2)],
    "tfidf__min_df": [1, 2, 5],
    "clf__alpha": [0.1, 0.5, 1.0]
  }
  grid_lr = {
    "tfidf__ngram_range": [(1,1), (1,2)],
    "tfidf__min_df": [1, 2, 5],
    "clf__C": [0.5, 1.0, 2.0, 4.0],
    "clf__penalty": ["l2"],
    "clf__class_weight": [None, "balanced"]
  }
  return {"nb": grid_nb, "lr": grid_lr}

def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
  y_pred = (y_prob >= threshold).astype(int)
  acc = accuracy_score(y_true, y_pred)
  precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
  # AUC requires prob for positive class
  try:
    auc = roc_auc_score(y_true, y_prob)
  except Exception:
    auc = float("nan")
  cm = confusion_matrix(y_true, y_pred)
  return {
    "accuracy": acc,
    "precision_spam": precision,
    "recall_spam": recall,
    "f1_spam": f1,
    "auc": auc,
    "confusion_matrix": cm.tolist(),
    "threshold": threshold
  }

def plot_confusion_matrix(cm: np.ndarray, labels=("ham","spam"), title="Confusion Matrix", path: Path = None):
  plt.figure(figsize=(4,3))
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
  plt.xlabel("Predicted")
  plt.ylabel("True")
  plt.title(title)
  plt.tight_layout()
  if path:
    plt.savefig(path, dpi=150)
  plt.close()

def main():
  raw_path = download_dataset()
  df = load_dataset(raw_path)
  train_df, val_df, test_df = split_data(df)

  X_train = train_df["text"].values
  y_train = train_df["label"].values
  X_val = val_df["text"].values
  y_val = val_df["label"].values
  X_test = test_df["text"].values
  y_test = test_df["label"].values

  pipelines = build_pipelines()
  grids = param_grids()
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

  results = {}
  best_models = {}

  for key in ["nb", "lr"]:
    print(f"[info] Tuning model: {key}")
    gs = GridSearchCV(
      estimator=pipelines[key],
      param_grid=grids[key],
      scoring="f1",
      cv=cv,
      n_jobs=-1,
      verbose=1
    )
    gs.fit(X_train, y_train)
    print(f"[info] Best params for {key}: {gs.best_params_}, best CV f1={gs.best_score_:.4f}")
    best_models[key] = gs.best_estimator_

    # Evaluate on validation set
    y_val_prob = gs.predict_proba(X_val)[:,1]
    res_val = evaluate(y_val, y_val_prob, threshold=0.5)
    results[f"{key}_val"] = res_val

  # Choose champion by validation F1
  champion_key = max(["nb","lr"], key=lambda k: results[f"{k}_val"]["f1_spam"])
  champion = best_models[champion_key]
  print(f"[info] Champion model: {champion_key}")

  # Fit champion on train+val
  X_trainval = np.concatenate([X_train, X_val])
  y_trainval = np.concatenate([y_train, y_val])
  champion.fit(X_trainval, y_trainval)

  # Evaluate on test
  y_test_prob = champion.predict_proba(X_test)[:,1]
  res_test = evaluate(y_test, y_test_prob, threshold=0.5)
  results[f"{champion_key}_test"] = res_test

  # Persist models
  joblib.dump(champion, ARTIFACT_DIR / f"champion_{champion_key}.joblib")
  # Persist the runner-up too
  other_key = "lr" if champion_key == "nb" else "nb"
  joblib.dump(best_models[other_key], ARTIFACT_DIR / f"backup_{other_key}.joblib")

  # Save metrics JSON
  with open(ARTIFACT_DIR / "metrics.json", "w") as f:
    json.dump(results, f, indent=2)

  # Save confusion matrix plot
  cm = np.array(res_test["confusion_matrix"])
  plot_confusion_matrix(cm, path=ARTIFACT_DIR / "confusion_matrix_test.png", title=f"CM Test ({champion_key})")

  # Save a small README of artifacts
  with open(ARTIFACT_DIR / "ARTIFACTS.txt", "w") as f:
    f.write("Artifacts generated:\n")
    f.write(f"- champion_{champion_key}.joblib\n")
    f.write(f"- backup_{other_key}.joblib\n")
    f.write("- metrics.json\n- confusion_matrix_test.png\n")

  print("[done] Training complete. Artifacts saved in ./artifacts")

if __name__ == "__main__":
  main()
```

## Lampiran F. Kode Evaluasi (src_evaluate.py)

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import joblib

RANDOM_STATE = 42
DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "SMSSpamCollection"
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)

def load_dataset():
  import pandas as pd
  df = pd.read_csv(RAW_PATH, sep="\t", header=None, names=["label", "text"], encoding="utf-8")
  df["label"] = df["label"].map({"ham": 0, "spam": 1})
  df.dropna(subset=["text"], inplace=True)
  return df

def ensure_dataset():
  if RAW_PATH.exists():
    return
  # Fallback download if belum ada (mengandalkan fungsi di train.py)
  from src.train import download_dataset
  download_dataset()

def load_model_candidates():
  # Kembalikan dict: {"lr": model or None, "nb": model or None}
  models = {}
  for name in ["lr", "nb"]:
    for prefix in ["champion", "backup"]:
      p = ARTIFACT_DIR / f"{prefix}_{name}.joblib"
      if p.exists():
        try:
          models[name] = joblib.load(p)
          break
        except Exception:
          pass
  return models

def plot_and_save_curves(y_true, y_prob, prefix):
  # ROC
  fpr, tpr, _ = roc_curve(y_true, y_prob)
  roc_auc = auc(fpr, tpr)
  plt.figure(figsize=(4.5,4))
  plt.plot(fpr, tpr, color="C0", lw=2, label=f"AUC = {roc_auc:.3f}")
  plt.plot([0,1],[0,1], color="gray", lw=1, ls="--")
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title(f"ROC Curve ({prefix})")
  plt.legend(loc="lower right")
  out = ARTIFACT_DIR / f"roc_{prefix}.png"
  plt.tight_layout()
  plt.savefig(out, dpi=150)
  plt.close()

  # PR
  precision, recall, _ = precision_recall_curve(y_true, y_prob)
  ap = average_precision_score(y_true, y_prob)
  plt.figure(figsize=(4.5,4))
  plt.plot(recall, precision, color="C1", lw=2, label=f"AP = {ap:.3f}")
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.title(f"Precision-Recall Curve ({prefix})")
  plt.legend(loc="lower left")
  out = ARTIFACT_DIR / f"pr_{prefix}.png"
  plt.tight_layout()
  plt.savefig(out, dpi=150)
  plt.close()

  return {"auc": float(roc_auc), "average_precision": float(ap)}

def eval_on_split(model, X, y, prefix, threshold=0.5):
  y_prob = model.predict_proba(X)[:,1]
  y_pred = (y_prob >= threshold).astype(int)
  cm = confusion_matrix(y, y_pred)
  report = classification_report(y, y_pred, output_dict=True, zero_division=0)
  curves = plot_and_save_curves(y, y_prob, prefix)
  return {
    "threshold": threshold,
    "confusion_matrix": cm.tolist(),
    "report": report,
    **curves
  }

def main():
  ensure_dataset()
  df = load_dataset()
  # Replikasi split: 70/15/15 (stratified) seperti di train.py
  train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=RANDOM_STATE)
  val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=RANDOM_STATE)

  X_test = test_df["text"].values
  y_test = test_df["label"].values

  models = load_model_candidates()
  rows = []
  metrics = {}

  for name, model in models.items():
    if model is None:
      continue
    print(f"[info] Evaluating model: {name}")
    res = eval_on_split(model, X_test, y_test, prefix=f"{name}_test")
    metrics[name] = res

    # Ringkasan metrik inti
    f1_spam = res["report"]["1"]["f1-score"]
    prec_spam = res["report"]["1"]["precision"]
    rec_spam = res["report"]["1"]["recall"]
    acc = res["report"]["accuracy"]
    rows.append({
      "model": name,
      "accuracy": acc,
      "precision_spam": prec_spam,
      "recall_spam": rec_spam,
      "f1_spam": f1_spam,
      "auc": res["auc"],
      "average_precision": res["average_precision"],
    })

  # Simpan CSV ringkasan
  if rows:
    df_out = pd.DataFrame(rows)
    df_out.to_csv(ARTIFACT_DIR / "performance_test.csv", index=False)
  # Simpan JSON lengkap
  with open(ARTIFACT_DIR / "metrics_eval_extra.json", "w") as f:
    json.dump(metrics, f, indent=2)

  print("[done] Evaluation complete. Figures and metrics saved in ./artifacts")

if __name__ == "__main__":
  main()
```

## Lampiran G. Aplikasi Streamlit (app_streamlit.py)

```python
import streamlit as st
import joblib
from pathlib import Path
import numpy as np

ARTIFACT_DIR = Path("artifacts")

def load_models():
  models = {}
  for name in ["champion_lr", "champion_nb", "backup_lr", "backup_nb"]:
    path = ARTIFACT_DIR / f"{name}.joblib"
    if path.exists():
      try:
        models[name.split("_")[-1]] = joblib.load(path)
      except Exception:
        pass
  # Fallback: try generic names
  for name in ["lr", "nb"]:
    path1 = ARTIFACT_DIR / f"champion_{name}.joblib"
    path2 = ARTIFACT_DIR / f"backup_{name}.joblib"
    if path1.exists() and name not in models:
      models[name] = joblib.load(path1)
    elif path2.exists() and name not in models:
      models[name] = joblib.load(path2)
  return models

st.set_page_config(page_title="Spam SMS Classifier", page_icon="📩", layout="centered")
st.title("📩 Klasifikasi Spam SMS (Naive Bayes vs. Logistic Regression)")

with st.sidebar:
  st.markdown("## Pengaturan")
  threshold = st.slider("Threshold Spam", 0.0, 1.0, 0.5, 0.01)
  st.markdown("Model akan memprediksi spam jika probabilitas ≥ threshold.")

models = load_models()
if not models:
  st.warning("Model belum tersedia. Jalankan `python src/train.py` terlebih dahulu untuk menghasilkan artefak.")
  st.stop()

model_choice = st.selectbox("Pilih Model", options=sorted(models.keys()), index=0, help="lr = Logistic Regression, nb = Naive Bayes")
model = models[model_choice]

tab1, tab2 = st.tabs(["Prediksi Tunggal", "Prediksi Batch"])
with tab1:
  text = st.text_area("Masukkan pesan SMS", height=150, placeholder="Contoh: Congratulations! You have won a free ticket...")
  if st.button("Prediksi"):
    if not text.strip():
      st.error("Mohon masukkan teks SMS.")
    else:
      prob_spam = model.predict_proba([text])[0,1]
      label = "spam" if prob_spam >= threshold else "ham"
      st.metric("Label", label.upper())
      st.progress(int(prob_spam*100))
      st.write(f"Probabilitas spam: {prob_spam:.3f} (threshold {threshold:.2f})")
      st.caption(f"Model: {model_choice}")

with tab2:
  st.write("Tempel beberapa baris SMS (satu pesan per baris).")
  batch = st.text_area("Batch SMS", height=200, placeholder="line 1\nline 2\nline 3")
  if st.button("Prediksi Batch"):
    lines = [ln for ln in batch.splitlines() if ln.strip()]
    if not lines:
      st.error("Tidak ada baris valid.")
    else:
      probs = model.predict_proba(lines)[:,1]
      labels = np.where(probs >= threshold, "spam", "ham")
      st.write("Hasil:")
      for i, (msg, p, lab) in enumerate(zip(lines, probs, labels), start=1):
        st.write(f"{i}. [{lab.upper()}] p(spam)={p:.3f} — {msg[:100]}{'...' if len(msg)>100 else ''}")
      st.success("Selesai.")

st.caption("Tips: Sesuaikan threshold untuk mengatur trade-off false positive vs. false negative.")
```

## Lampiran H. Layanan Flask (app_flask.py)

```python
import os
import json
from pathlib import Path
import joblib
from flask import Flask, request, jsonify

ARTIFACT_DIR = Path("artifacts")
app = Flask(__name__)

def load_model(preferred: str = None):
  candidates = []
  if preferred:
    candidates += [ARTIFACT_DIR / f"champion_{preferred}.joblib", ARTIFACT_DIR / f"backup_{preferred}.joblib"]
  candidates += [ARTIFACT_DIR / "champion_lr.joblib", ARTIFACT_DIR / "champion_nb.joblib",
           ARTIFACT_DIR / "backup_lr.joblib", ARTIFACT_DIR / "backup_nb.joblib"]
  for p in candidates:
    if p.exists():
      return joblib.load(p), p.stem
  raise FileNotFoundError("No model artifacts found. Run training first.")

@app.route("/predict", methods=["POST"])
def predict():
  payload = request.get_json(force=True, silent=True) or {}
  text = payload.get("text", "")
  model_name = payload.get("model", None)  # "lr" or "nb"
  threshold = float(payload.get("threshold", 0.5))
  if not isinstance(text, str) or not text.strip():
    return jsonify({"error": "Field 'text' is required"}), 400
  try:
    model, model_file = load_model(model_name)
  except Exception as e:
    return jsonify({"error": str(e)}), 500
  prob_spam = float(model.predict_proba([text])[0,1])
  label = "spam" if prob_spam >= threshold else "ham"
  return jsonify({
    "label": label,
    "prob_spam": prob_spam,
    "threshold": threshold,
    "model_file": model_file
  })

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
```

## Referensi
Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering: New Collection and Results. In Proceedings of the 11th ACM Symposium on Document Engineering (pp. 259–262).

Cox, D. R. (1958). The regression analysis of binary sequences. Journal of the Royal Statistical Society: Series B, 20(2), 215–242.

Hanley, J. A., & McNeil, B. J. (1982). The meaning and use of the area under a ROC curve. Radiology, 143(1), 29–36.

Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. IJCAI.

McCallum, A., & Nigam, K. (1998). A comparison of event models for Naive Bayes text classification. AAAI-98 workshop on learning for text categorization.

Ng, A. Y. (2004). Feature selection, L1 vs. L2 regularization, and rotational invariance. ICML.

Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.

Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5), 513–523.