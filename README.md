# LPDP Twitter Sentiment Analysis

Refactored and modularised Indonesian sentiment analysis pipeline utilizing a Random Forest Classifier and the InSet Lexicon dataset.

## 📂 Project Structure

```text
├── .agents/          # Workspace agent configurations & guidelines
├── notebooks/        # Jupyter notebooks (original source files)
├── src/
│   ├── preprocessing.py  # Indonesian cleaning, slang normalization, and Sastrawi stemming
│   ├── lexicon.py        # TSV lexicon loader, word conflict resolution, and sentiment scoring
│   ├── eda.py            # Wordclouds, distributions, and trend analysis plots
│   ├── model.py          # TF-IDF vectorization, Stratified CV training, and tuning
│   └── inference.py      # Predictor class and CLI inference utility
├── pyproject.toml    # Project dependencies and configurations
├── uv.lock           # Dependency lockfile
└── main.py           # End-to-end master pipeline orchestrator
```

---

## 🚀 How to Run

Ensure that dependency packages are synced using the `uv` package manager:
```bash
uv sync
```

### 1. Run the Pipeline with Hyperparameter Search

Runs text cleaning, labeling, EDA figure generation, base classifier evaluation, and 150 RandomizedSearchCV iterations:
```bash
uv run python main.py
```

### 2. Run the Pipeline Quickly (Skipping Tuning)

Skips hyperparameter search to speed up execution (~3.5 minutes due to Indonesian stemming):
```bash
uv run python main.py --skip-tuning
```

### 3. Run Inference via CLI

Classifies sentiment categories and confidence scores of custom texts using the serialized best model:
```bash
uv run python src/inference.py --text "Alhamdulillah lolos beasiswa LPDP! Sangat membantu."
```
