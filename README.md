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

### Text Preprocessing Pipeline

This project includes a configurable text preprocessing function (`clean_text`) designed for linguistically informed NLP and corpus analysis workflows.

Rather than applying fixed cleaning rules, the pipeline allows individual preprocessing steps to be toggled on or off depending on the task (e.g., frequency analysis, collocation extraction, discourse analysis).

---

### `clean_text` Configuration Options

The `clean_text` function accepts a dictionary of preprocessing options via the `steps` argument. Any omitted options fall back to sensible defaults.

#### Options

- **`lowercase`**  
  Converts all letters to lowercase when `True`. This is useful for frequency-based analyses where capitalization differences (e.g., *“World”* vs. *“world”*) should not be treated as distinct types.

- **`remove_punct`**  
  Removes punctuation when `True`. This helps ensure that tokens attached to punctuation (e.g., *“word,”*) are treated the same as their unpunctuated forms.

- **`keep_apostrophes`**  
  When punctuation removal is enabled, preserves apostrophes when `True`. This is useful for maintaining contractions (e.g., *“don’t”*), which may lose linguistic information if apostrophes are removed.

- **`keep_hyphens`**  
  When punctuation removal is enabled, preserves hyphens when `True`. This helps conserve hyphenated multiword expressions (e.g., *“well-being”*), which often function as single semantic units.

- **`normalize_whitespace`**  
  Collapses sequences of multiple spaces into a single space when `True`, producing clean, model-ready text.

- **`preserve_newlines`**  
  When whitespace normalization is enabled, preserves newline characters when `True`. This allows paragraph or line boundaries to be retained for discourse-level or formatting-sensitive analyses.

---

### Example Usage

```python
from preprocess import clean_text

cleaned_text = clean_text(
    raw_text,
    steps={
        "lowercase": True,
        "remove_punct": True,
        "keep_apostrophes": True,
        "keep_hyphens": True,
        "normalize_whitespace": True,
        "preserve_newlines": False,
    }
)
---

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
