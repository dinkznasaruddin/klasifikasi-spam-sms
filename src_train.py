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
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
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
from scipy import sparse
import re

RANDOM_STATE = 42
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

UCI_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


class LexicalFeatures(BaseEstimator, TransformerMixin):
    """Simple regex-based binary features to help catch Indonesian spam cues.

    Features (columns):
      0: has_url
      1: has_shortener
      2: has_money_token (rp/idr/$ or numeric currency-like)
      3: has_bank_keyword (bca,bni,bri,mandiri,cimb,ovo,dana,gopay,wa.me)
      4: has_action_keyword (verifikasi, blokir, aktif, klik, konfirmasi, bayar, klaim)
    """

    def __init__(self):
        self.re_url = re.compile(r"(https?://|www\.|wa\.me)", re.I)
        self.re_short = re.compile(r"(bit\.ly|goo\.gl|s\.id|tinyurl|t\.co|cutt\.ly)", re.I)
        self.re_money = re.compile(r"(\brp\b|\bidr\b|\$|\d{2,3}[\.\s]?\d{3})", re.I)
        self.re_bank = re.compile(r"\b(bca|bni|bri|mandiri|cimb|ovo|dana|gopay)\b", re.I)
        self.re_action = re.compile(r"\b(verifikasi|blokir|aktif|konfirmasi|klaim|claim|bayar|klik)\b", re.I)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            t = str(text)
            has_url = 1 if self.re_url.search(t) else 0
            has_short = 1 if self.re_short.search(t) else 0
            has_money = 1 if self.re_money.search(t) else 0
            has_bank = 1 if self.re_bank.search(t) else 0
            has_action = 1 if self.re_action.search(t) else 0
            rows.append([has_url, has_short, has_money, has_bank, has_action])
        mat = np.asarray(rows, dtype=np.float32)
        return sparse.csr_matrix(mat)

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
    features_nb = FeatureUnion([
        ("tfidf", TfidfVectorizer()),
        ("lex", LexicalFeatures()),
    ])
    features_lr = FeatureUnion([
        ("tfidf", TfidfVectorizer()),
        ("lex", LexicalFeatures()),
    ])
    pipe_nb = Pipeline(steps=[
        ("features", features_nb),
        ("clf", MultinomialNB())
    ])
    pipe_lr = Pipeline(steps=[
        ("features", features_lr),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))
    ])
    return {"nb": pipe_nb, "lr": pipe_lr}

def param_grids() -> Dict[str, Dict]:
    # NB (word features); stop_words dimatikan agar tidak salah untuk bahasa non-Inggris
    grid_nb = {
        "features__tfidf__analyzer": ["word"],
        "features__tfidf__stop_words": [None, "english"],
        "features__tfidf__ngram_range": [(1,1), (1,2)],
        "features__tfidf__min_df": [1, 2],
        "clf__alpha": [0.1, 0.5]
    }
    # LR: dua skenario grid — word-level dan char-level (lebih tangguh lintas bahasa/obfuscation)
    grid_lr = [
        {
            "features__tfidf__analyzer": ["word"],
            "features__tfidf__stop_words": [None, "english"],
            "features__tfidf__ngram_range": [(1,1), (1,2)],
            "features__tfidf__min_df": [1, 2],
            "clf__C": [0.5, 1.0, 2.0],
            "clf__penalty": ["l2"],
            "clf__class_weight": [None, "balanced"]
        },
        {
            "features__tfidf__analyzer": ["char_wb"],
            "features__tfidf__ngram_range": [(3,5)],
            "features__tfidf__min_df": [1],
            "clf__C": [1.0, 2.0],
            "clf__penalty": ["l2"],
            "clf__class_weight": [None, "balanced"]
        }
    ]
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