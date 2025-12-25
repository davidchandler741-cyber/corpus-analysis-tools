import os
import re
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import clean_text

def load_documents(input_folder: str):
    """Load raw + cleaned documents from .txt files (excluding CLEAN_ files)."""
    filenames = []
    raw_docs = []
    cleaned_docs = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt") and not filename.startswith("CLEAN_"):
            path = os.path.join(input_folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()

            filenames.append(filename)
            raw_docs.append(raw)
            cleaned_docs.append(clean_text(raw))

    return filenames, raw_docs, cleaned_docs


def spacy_features(nlp, raw_text: str) -> dict:
    """Extract a small set of spaCy-based features from raw text."""
    doc = nlp(raw_text)
    return {
        "n_tokens_spacy": len(doc),
        "n_stopwords_spacy": sum(tok.is_stop for tok in doc),
        "n_adjectives_spacy": sum(tok.pos_ == "ADJ" for tok in doc),
        "n_verbs_spacy": sum(tok.pos_ == "VERB" for tok in doc),
    }


def build_feature_dataframe(filenames, raw_docs, cleaned_docs, nlp) -> pd.DataFrame:
    """Build a DataFrame with raw + NLP features."""
    rows = []

    for filename, raw, cleaned in zip(filenames, raw_docs, cleaned_docs):
        text_length = len(raw)
        num_lines = raw.count("\n") + 1

        feats = spacy_features(nlp, raw)

        row = {
            "file": filename,
            "raw_length": text_length,
            "num_lines": num_lines,
            "cleaned_text": cleaned,
            **feats,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Derived ratios (per-token normalization)
    df["pct_stopwords_spacy"] = df["n_stopwords_spacy"] / df["n_tokens_spacy"]
    df["pct_adjectives_spacy"] = df["n_adjectives_spacy"] / df["n_tokens_spacy"]
    df["pct_verbs_spacy"] = df["n_verbs_spacy"] / df["n_tokens_spacy"]

    return df


def build_tfidf_dataframe(cleaned_docs) -> tuple[pd.DataFrame, TfidfVectorizer]:
    """Create a TF-IDF matrix as a DataFrame."""
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned_docs)
    tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    return tfidf_df, vectorizer


def main():
    input_folder = "data_in"
    output_folder = "data_out"
    os.makedirs(output_folder, exist_ok=True)

    nlp = spacy.load("en_core_web_sm")

    filenames, raw_docs, cleaned_docs = load_documents(input_folder)

    # 1) spaCy + structural features
    features_df = build_feature_dataframe(filenames, raw_docs, cleaned_docs, nlp)
    features_path = os.path.join(output_folder, "review_features.csv")
    features_df.to_csv(features_path, index=False, encoding="utf-8")
    print(f"Saved: {features_path}")

    # 2) TF-IDF matrix
    tfidf_df, _ = build_tfidf_dataframe(cleaned_docs)
    tfidf_path = os.path.join(output_folder, "review_tfidf.csv")
    tfidf_df.to_csv(tfidf_path, index=False, encoding="utf-8")
    print(f"Saved: {tfidf_path}")

    # Quick glance
    print("\nFeature table preview:")
    print(features_df[["file", "pct_stopwords_spacy", "pct_adjectives_spacy", "pct_verbs_spacy"]])


if __name__ == "__main__":
    main()
