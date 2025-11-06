# Glosarium Istilah (Versi Mudah Dimengerti)

Tujuan: membantu pembaca memahami istilah penting di presentasi/paper dengan bahasa sederhana dan contoh singkat.

## Dataset & Teks
- SMS Spam Collection (UCI): Kumpulan data SMS berlabel “ham” (normal) atau “spam”, dari UCI Machine Learning Repository.
- Ham: Pesan normal/legit (bukan spam). Contoh: "Aku otw ya."
- Spam: Pesan tak diinginkan/penipuan/promosi agresif. Contoh: "GRATIS! Klik link ini untuk hadiah."
- Phishing: Upaya menipu untuk mencuri data (mis. minta OTP/akun) lewat pesan.
- Token: Potongan teks (biasanya kata) setelah dipisah spasi/tanda baca.
- Stopword: Kata-kata sangat umum yang sering diabaikan (the, and, of, dll) karena kurang informatif.
- N-gram: Urutan n kata berurutan. Unigram = 1 kata (“free”), bigram = 2 kata (“claim now”).
- TSV (tab-separated values): Format file teks dengan kolom dipisah tanda tab.

## Fitur & Pratata Kelola (Preprocessing)
- Bag-of-Words (BoW): Representasi teks sebagai hitungan kata tanpa memperhatikan urutan panjang.
- TF–IDF: Cara memberi bobot kata. Kata sering di dokumen tapi jarang di seluruh korpus dapat bobot tinggi. Membantu menonjolkan kata yang “unik”.
- Normalisasi (lowercasing): Mengubah huruf menjadi kecil semua agar “Free” dan “free” dianggap sama.
- Vektor Fitur: Angka-angka yang mewakili teks agar bisa diproses model ML.

## Model & Hiperparameter
- Multinomial Naive Bayes (MNB): Model probabilistik sederhana untuk teks. Cepat dan efektif sebagai baseline.
- Logistic Regression (LR): Model klasifikasi yang menghitung peluang sebuah pesan adalah spam (0–1), dengan regularisasi agar tidak overfitting.
- Regularisasi (L2/L1): Hukuman pada besar bobot model agar model tidak terlalu “menghafal” data latih.
- Alpha (α) pada MNB: Derajat smoothing (semakin besar, semakin “halus” estimasi probabilitas kata langka).
- C pada LR: Kebalikan kekuatan regularisasi. C besar → penalti kecil (lebih fleksibel), C kecil → penalti besar (lebih sederhana).
- class_weight=balanced: Menyeimbangkan pengaruh kelas saat kelas tidak seimbang (spam lebih sedikit dari ham).
- Threshold: Ambang probabilitas untuk memutuskan “spam” (mis. ≥ 0,5 → spam). Bisa diubah sesuai biaya bisnis.
- Kalibrasi Probabilitas: Menyesuaikan skor model agar probabilitas lebih “jujur” (mis. Platt scaling, isotonic).

## Validasi & Pelatihan
- Cross-Validation (k-fold): Membagi data menjadi k bagian untuk evaluasi bergiliran agar penilaian lebih stabil.
- Stratified: Pembagian yang menjaga perbandingan proporsi kelas sama di tiap lipatan (fold).
- GridSearchCV: Mencoba kombinasi hiperparameter yang berbeda untuk mencari yang terbaik.
- Seed (random_state): Angka acuan agar hasil eksperimen bisa diulang (reproducible).

## Metrik Evaluasi
- Accuracy: Proporsi prediksi benar dari semua kasus. Tidak cukup saat kelas tidak seimbang.
- Precision (untuk spam): Dari semua yang diprediksi spam, berapa yang benar-benar spam? Tinggi precision → sedikit false positive.
- Recall (untuk spam): Dari semua spam sebenarnya, berapa yang berhasil ditangkap? Tinggi recall → sedikit false negative.
- F1-score: Rata-rata harmonik precision dan recall. Cocok saat butuh keseimbangan keduanya.
  - Rumus singkat: F1 = 2 × (Precision × Recall) / (Precision + Recall)
- Confusion Matrix: Tabel ringkas benar/salah per kelas.
  - True Positive (TP): Spam yang terdeteksi spam.
  - False Positive (FP): Ham yang keliru disebut spam.
  - False Negative (FN): Spam yang lolos (tersebut ham).
  - True Negative (TN): Ham yang benar disebut ham.
- ROC Curve & AUC: Kurva trade-off TPR vs FPR saat threshold diubah; AUC mendekati 1 artinya sangat baik.
- Precision–Recall (PR) Curve & AP: Lebih informatif untuk kelas jarang (spam sedikit). AP (average precision) serupa AUC di PR space.

## Prototipe & Integrasi
- Streamlit: Aplikasi web cepat untuk demo prediksi (input pesan → probabilitas spam → label).
- Flask API: Layanan HTTP (endpoint /predict) untuk integrasi sistem lain.
- Inferensi: Proses memberikan prediksi dari model terlatih.
- Artefak Model: File hasil pelatihan (mis. .joblib) yang berisi pipeline TF–IDF + model.

## Operasional (MLOps)
- Monitoring: Mengamati metrik, distribusi input, dan keluhan untuk mendeteksi masalah.
- Drift: Perubahan pola data (mis. gaya baru spam) yang membuat model menurun performanya.
- Retraining: Melatih ulang model secara berkala atau ketika terdeteksi drift.
- Threshold Tuning: Menyetel ambang berdasarkan biaya/kebijakan (lebih sensitif atau lebih ketat).

## Contoh Mini (Threshold)
- Model memberi p(spam) = 0,62. Jika threshold = 0,50 → spam. Jika threshold = 0,70 → ham.
- Atur threshold sesuai prioritas: hindari lolosnya spam (naikkan recall, threshold lebih rendah) vs hindari salah tandai pesan sah (naikkan precision, threshold lebih tinggi).
