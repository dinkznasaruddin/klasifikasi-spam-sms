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