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