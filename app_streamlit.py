import streamlit as st
import joblib
from pathlib import Path
import numpy as np
import re

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

st.set_page_config(page_title="TUGAS KLP 4 (Artificial Intelligent) — Spam SMS Classifier", page_icon="📩", layout="centered")
st.title("TUGAS KLP 4 (Artificial Intelligent)")
st.markdown("### 📩 Klasifikasi Spam SMS (Naive Bayes vs. Logistic Regression)")

with st.sidebar:
    st.markdown("### TUGAS KLP 4 (Artificial Intelligent)")
    st.markdown("## Pengaturan")
    threshold = st.slider(
        "Threshold Spam",
        0.0,
        1.0,
        0.5,
        0.01,
        help=(
            "Ambang keputusan. Jika probabilitas spam ≥ threshold maka pesan diklasifikasikan sebagai SPAM;"
            " jika < threshold maka HAM. Turunkan untuk lebih sensitif (menangkap lebih banyak spam),"
            " naikkan untuk lebih ketat (mengurangi false positive)."
        ),
    )
    with st.expander("Apa itu Threshold Spam?", expanded=False):
        st.markdown(
            """
            Threshold adalah batas probabilitas untuk mengambil keputusan SPAM/HAM.

            - Aturan keputusan: jika p(spam) ≥ threshold → SPAM, jika p(spam) < threshold → HAM.
            - Pengaruh pengaturan:
              - Threshold lebih rendah (mis. 0.45): recall SPAM naik, tapi false positive bisa bertambah.
              - Threshold lebih tinggi (mis. 0.60): lebih ketat, false positive turun, tapi sebagian SPAM bisa lolos.
            - Rekomendasi awal: 0.45 – 0.55. Sesuaikan sesuai kebutuhan.
            """
        )
    use_boost = st.checkbox(
        "Aktifkan booster heuristik (URL/bank/aksi)",
        value=False,
        help=(
            "Jika diaktifkan, probabilitas akan dinaikkan sedikit untuk pesan yang mengandung pola kuat seperti URL,"
            " shortener (bit.ly, t.co), kata bank (BCA, BNI, BRI, Mandiri), atau kata aksi (verifikasi, blokir, klik)."
        ),
    )

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

            # Optional heuristic boost for Indonesian spam cues
            prob_used = prob_spam
            detail = ""
            if use_boost:
                t = text.lower()
                has_url = bool(re.search(r"https?://|www\.|wa\.me", t))
                has_short = bool(re.search(r"bit\.ly|goo\.gl|s\.id|tinyurl|t\.co|cutt\.ly", t))
                has_bank = bool(re.search(r"\b(bca|bni|bri|mandiri|cimb|ovo|dana|gopay)\b", t))
                has_action = bool(re.search(r"\b(verifikasi|blokir|aktif|konfirmasi|klaim|claim|bayar|klik)\b", t))
                boost = 0.0
                if has_url and (has_short or has_bank or has_action):
                    boost = 0.35
                elif has_url or has_short or has_action:
                    boost = 0.20
                prob_used = min(1.0, prob_spam + boost)
                detail = f" | booster +{boost:.2f}"

            label = "spam" if prob_used >= threshold else "ham"
            st.metric("Label", label.upper())
            st.progress(int(prob_used*100))
            st.write(f"Probabilitas spam: {prob_spam:.3f} → dipakai {prob_used:.3f}{detail} (threshold {threshold:.2f})")
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
            probs_used = probs.copy()
            if use_boost:
                adjusted = []
                for msg, p in zip(lines, probs):
                    t = msg.lower()
                    has_url = bool(re.search(r"https?://|www\.|wa\.me", t))
                    has_short = bool(re.search(r"bit\.ly|goo\.gl|s\.id|tinyurl|t\.co|cutt\.ly", t))
                    has_bank = bool(re.search(r"\b(bca|bni|bri|mandiri|cimb|ovo|dana|gopay)\b", t))
                    has_action = bool(re.search(r"\b(verifikasi|blokir|aktif|konfirmasi|klaim|claim|bayar|klik)\b", t))
                    boost = 0.35 if (has_url and (has_short or has_bank or has_action)) else (0.20 if (has_url or has_short or has_action) else 0.0)
                    adjusted.append(min(1.0, p + boost))
                probs_used = np.array(adjusted)
            labels = np.where(probs_used >= threshold, "spam", "ham")
            st.write("Hasil:")
            for i, (msg, p0, pu, lab) in enumerate(zip(lines, probs, probs_used, labels), start=1):
                extra = "" if abs(pu - p0) < 1e-6 else f" (→ {pu:.3f} dg booster)"
                st.write(f"{i}. [{lab.upper()}] p(spam)={p0:.3f}{extra} — {msg[:100]}{'...' if len(msg)>100 else ''}")
            st.success("Selesai.")

st.caption("Tips: Sesuaikan threshold untuk mengatur trade-off false positive vs. false negative.")