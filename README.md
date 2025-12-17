# Review NLP Analysis Tool

A lightweight Python NLP pipeline that analyzes short text reviews using
spaCy and classic NLP techniques (TF-IDF), producing structured features
for data analysis or downstream machine learning.

This project demonstrates practical NLP feature engineering, pandas-based
analysis, and clean project organization.

---

## 📁 Project Structure

review_nlp_tool/

│

├── data_in/ # Input .txt review files

├── data_out/ # Generated CSV outputs

├── main.py # NLP pipeline entry point

├── requirements.txt # Python dependencies

└── README.md


---

## 🔧 Features

### Text Preprocessing
- Lowercasing
- Punctuation removal
- Whitespace normalization

### spaCy Linguistic Features
For each review:
- Number of tokens
- Number of stopwords
- Number of adjectives
- Number of verbs
- Normalized ratios (per token)

### Classic NLP (TF-IDF)
- Converts cleaned text into a TF-IDF matrix
- Outputs a document–term matrix suitable for ML models

---

## How to Run

1. Place `.txt` review files in `data_in/`
2. Activate your virtual environment
3. Run:

```bash
py main.py

## 📊 Outputs

Generated in `data_out/`:

- **`review_features.csv`**
  - Structural + linguistic features per review

- **`review_tfidf.csv`**
  - TF-IDF document–term matrix

---

## 🧠 Skills Demonstrated

- Python scripting and file I/O
- NLP preprocessing pipelines
- spaCy feature extraction
- TF-IDF vectorization (scikit-learn)
- pandas-based feature engineering
- Reproducible project structure

---

## 📌 Potential Extensions

- Sentiment classification
- Topic modeling
- Review clustering
- Integration with machine learning models
- API or web interface

---

## 🧑‍💻 Author

**David**  
Applied Linguistics (Computational / NLP focus)
