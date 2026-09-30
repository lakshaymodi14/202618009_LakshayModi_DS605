# DS605 Lab 6 — Feature Extraction & ML on Image and Text Data

Converting raw asphalt images and raw email data into numeric features, then classifying with traditional ML (no CNNs, no pretrained embeddings).

## Datasets
- **Images**: [Asphalt Crack Dataset](https://data.mendeley.com/) — 400 images (200 Crack / 200 NonCrack), 448×448 RGB.
- **Text**: [Email Spam Classification Dataset](https://www.kaggle.com/) — 5,172 emails, provided pre-vectorized as word-count columns (3000 words + label).

Datasets aren't included in this repo (too large / not ours to redistribute). Download them and place as:
```
data/images/Cracks/*.jpg
data/images/NonCracks/*.jpg
data/emails.csv
```

## Part A — Images
Pipeline: resize (256×256) → grayscale → intensity stats (mean, std, percentiles, dark/bright ratios) → Canny edges (blur + threshold tuned **on training data only**, to avoid leakage) → 11-feature table → Random Forest / SVM / Logistic Regression.

**Result**: Random Forest, 95% accuracy / 0.95 F1 on the held-out split. Feature importances showed edge features drove ~40% of decisions, with brightness contributing the rest, i.e. the model is reading crack *structure*, not just exploiting lighting differences between photo batches.

Since 400 images makes any single split noisy, we also ran 5-fold CV: **mean F1 = 0.89** (range 0.67–1.00), which we treat as the more honest number.

## Part B — Text
The dataset arrived already vectorized (word-count columns), so this used that representation directly rather than running CountVectorizer/TF-IDF ourselves — noted here for transparency. Naive Bayes and Logistic Regression were trained and compared.

**Result**: Logistic Regression, mean F1 = 0.944 (5-fold CV), clearly ahead of Naive Bayes (0.904).

## Part C — Improvements (tried several, kept what worked)

| Change | Outcome |
|---|---|
| Naive Canny (no blur) → tuned Canny (blur + thresholds picked from train data) | 63.8% → 95.0% accuracy — the headline win |
| Multiscale edge density (2nd Canny pass at heavier blur) | CV F1 0.890 → 0.898, rescued our weakest fold |
| Skewness / kurtosis / variance features | **Hurt** — CV F1 0.890 → 0.869, redundant with existing contrast stats |
| Image resolution 128→256→448 | Accuracy flat (~0.95); time scaled 6× — kept 256 |
| Vocabulary limiting (3000 → 500 words) | Matched full-vocab F1, ~85% less training time |
| Word-length / total-count features | **Hurt slightly** — info was already implicit in word counts |
| PCA (50–500 components) | **Hurt** — variance-maximizing components ≠ spam-discriminating words |
| Regularization tuning (C=1 → C=0.01) | CV F1 0.944 → 0.958 — best single text improvement |

## Setup
```bash
pip install -r requirements.txt
jupyter notebook notebooks/Lab06.ipynb
```
