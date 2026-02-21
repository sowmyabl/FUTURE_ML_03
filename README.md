# Resume / Candidate Screening System

A Python-based machine learning project that automatically screens and ranks
resumes against a given job description. The system cleans and parses resume
text, extracts skills, computes similarity scores, identifies skill gaps, and
provides a simple Streamlit web interface for evaluation and visualization.

## Features

- **Text preprocessing:** stopword removal, lowercasing, punctuation stripping,
  tokenization and lemmatization using spaCy.
- **Skill extraction:** matches a predefined vocabulary of technical skills
  against resume content.
- **Scoring:** TF-IDF vectorization (with n-grams) and cosine similarity to
  compute a percentage score between resume and job description.
- **Ranking:** candidates are ranked in descending order of similarity score.
- **Skill gap identification:** lists required skills missing from each resume.
- **Visualizations:** bar charts of scores, pie charts of skill match percentages,
  horizontal bar graphs of missing skills (Matplotlib only).
- **Streamlit UI:** modern dark-blue theme, yellow title banner, file upload and
  job description input, with an "Analyze & Rank" button.

## Technologies Used

- Python 3.x
- [spaCy](https://spacy.io) for NLP preprocessing
- [scikit-learn](https://scikit-learn.org) for TF-IDF and cosine similarity
- pandas and NumPy for data handling
- Matplotlib for plots
- Streamlit for the web application
- PyPDF2 for PDF resume parsing

## Folder Structure

```
project-root/
├── data/                 # place raw resumes or job descriptions here if needed
├── app.py                # Streamlit application
├── model.py              # TF-IDF vectorization and scoring logic
├── preprocessing.py      # text cleaning and tokenization
├── ranking.py            # ranking, scoring and plotting helpers
├── utils.py              # resume loading and skill utilities
├── requirements.txt      # third-party dependencies
└── README.md             # this document
```

## Getting Started

1. **Clone the repository** (or copy the files into a workspace):
   ```bash
   git clone <repo-url>
   cd project-root
   ```

2. **Create a Python environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # download the spaCy model
   python -m spacy download en_core_web_sm
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

   A browser window will open with the resume screening interface. Upload one or
   more resumes in .txt or .pdf format, paste or type a job description, and
   click **Analyze & Rank**.

## Sample Output Explanation

After analysis, the application displays:

- A **ranking table** listing each candidate, their similarity score,
  matched skills and missing skills.
- A **bar chart** showing scores for all applicants.
- For each candidate:
  - A **pie chart** indicating the percentage of required skills that were
    found versus missing.
  - A **horizontal bar chart** listing missing skills by name.

The scores are computed by converting text to a TF-IDF vector space and
computing cosine similarity with the job description. Preprocessing ensures that
common words are ignored and terms are lemmatized, so variations of the same
word still contribute to similarity.

## Improving Accuracy (Optional)

- Expand the skills vocabulary and use fuzzy matching for skill extraction.
- Experiment with n-gram ranges in the TF-IDF vectorizer or other feature
  engineering techniques.
- Train a supervised classifier such as Logistic Regression or an SVM on a
  labeled dataset of resumes (accepted/rejected) to predict suitability.
- Incorporate more advanced NLP features like word embeddings (Word2Vec,
  BERT) for semantic similarity.

## Code Overview

Each Python module is documented with comments explaining the purpose of
functions and the reason for design choices (e.g. use of TF-IDF for
term-weighting, cosine similarity for vector comparison, and spaCy for
preprocessing). The modular structure keeps preprocessing, modeling, ranking,
and utilities separate for clarity and maintainability.

---

Feel free to extend or adapt this project for your own resume screening
workflows. Contributions are welcome!
