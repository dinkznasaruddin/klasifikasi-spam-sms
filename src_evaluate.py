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
    # Import dari modul skrip pelatihan yang ada di proyek ini
    from src_train import download_dataset
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