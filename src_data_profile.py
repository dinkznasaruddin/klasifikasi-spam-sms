from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from src_train import download_dataset, load_dataset

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)


def tokenize_len(s: str):
    # simple whitespace tokenization
    return len(s.split())


def main():
    raw_path = download_dataset()
    df = load_dataset(raw_path)

    # Basic counts
    n_total = len(df)
    n_spam = int((df["label"] == 1).sum())
    n_ham = int((df["label"] == 0).sum())
    pct_spam = n_spam / n_total

    # Lengths
    df["len_tokens"] = df["text"].map(tokenize_len)
    df["len_chars"] = df["text"].str.len()

    def series_stats(s: pd.Series):
        return {
            "mean": float(np.mean(s)),
            "median": float(np.median(s)),
            "std": float(np.std(s, ddof=1)),
            "min": float(np.min(s)),
            "max": float(np.max(s)),
        }

    stats_all = series_stats(df["len_tokens"]) | {"chars": series_stats(df["len_chars"]) }
    stats_spam = series_stats(df.loc[df["label"]==1, "len_tokens"]) | {"chars": series_stats(df.loc[df["label"]==1, "len_chars"]) }
    stats_ham = series_stats(df.loc[df["label"]==0, "len_tokens"]) | {"chars": series_stats(df.loc[df["label"]==0, "len_chars"]) }

    # Top tokens per class (stopwords removed)
    vec = CountVectorizer(stop_words="english", ngram_range=(1,1), min_df=2)
    X = vec.fit_transform(df["text"])  # all docs
    vocab = np.array(vec.get_feature_names_out())

    top_k = 20
    # spam mask
    spam_idx = df["label"].values == 1
    ham_idx = ~spam_idx

    spam_counts = np.asarray(X[spam_idx].sum(axis=0)).ravel()
    ham_counts = np.asarray(X[ham_idx].sum(axis=0)).ravel()

    top_spam_idx = np.argsort(spam_counts)[-top_k:][::-1]
    top_ham_idx = np.argsort(ham_counts)[-top_k:][::-1]

    top_spam = [(vocab[i], int(spam_counts[i])) for i in top_spam_idx]
    top_ham = [(vocab[i], int(ham_counts[i])) for i in top_ham_idx]

    # Plots
    sns.set_style("whitegrid")

    # Histogram of token lengths
    plt.figure(figsize=(6,4))
    sns.histplot(df, x="len_tokens", hue=df["label"].map({0:"ham",1:"spam"}), bins=40, stat="density", common_norm=False)
    plt.xlabel("Panjang pesan (jumlah token)")
    plt.ylabel("Kepadatan")
    plt.title("Distribusi Panjang Pesan (Token)")
    plt.tight_layout()
    (ARTIFACT_DIR / "len_hist.png").unlink(missing_ok=True)
    plt.savefig(ARTIFACT_DIR / "len_hist.png", dpi=150)
    plt.close()

    # Bar plots top tokens
    def bar_plot(pairs, title, path):
        terms, counts = zip(*pairs)
        plt.figure(figsize=(7.2,4.8))
        sns.barplot(x=list(counts), y=list(terms), orient="h", color="#4C78A8")
        plt.xlabel("Frekuensi")
        plt.ylabel("Term")
        plt.title(title)
        plt.tight_layout()
        Path(path).unlink(missing_ok=True)
        plt.savefig(path, dpi=150)
        plt.close()

    bar_plot(top_spam, "Top Token Spam (Unigram)", ARTIFACT_DIR / "top_tokens_spam.png")
    bar_plot(top_ham, "Top Token Ham (Unigram)", ARTIFACT_DIR / "top_tokens_ham.png")

    # Save JSON summary
    summary = {
        "n_total": n_total,
        "n_spam": n_spam,
        "n_ham": n_ham,
        "pct_spam": pct_spam,
        "len_tokens": {"all": stats_all, "spam": stats_spam, "ham": stats_ham},
        "top_spam": top_spam,
        "top_ham": top_ham,
    }
    with open(ARTIFACT_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[done] Dataset profile artifacts saved to ./artifacts")


if __name__ == "__main__":
    main()
