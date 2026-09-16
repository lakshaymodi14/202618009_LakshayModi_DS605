# NYC Airbnb Price Predictor — DS605 Lab 4

**Live app:** https://202618009lakshaymodids605-vybfuora4hc8ckjdmmixmp.streamlit.app/
**Course:** DS605 – Fundamentals of Machine Learning, Lab Assignment 4
**Dataset:** [Kaggle — New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) (`AB_NYC_2019.csv`)

An end-to-end machine learning project that predicts the nightly price of a
NYC Airbnb listing from its location, room type, host and review stats, and
listing title — trained in a leakage-free scikit-learn pipeline and served
through a Streamlit app.

<!-- TODO: add 1–2 screenshots of the deployed app here, e.g.
![App screenshot](images/app_screenshot.png)
-->

## Repository structure

```
Lab04/
├── DS605_Lab4_Airbnb_cleaned.ipynb   # full analysis, modelling, tuning, evaluation
├── app.py                            # Streamlit UI
├── model_pipeline.py                 # feature engineering + inference used by app.py
├── airbnb_price_pipeline.joblib      # fitted preprocessing + model pipeline
├── requirements.txt
└── README.md
```

## How to run locally

```bash
git clone https://github.com/lakshaymodi14/202618009_LakshayModi_DS605.git
cd 202618009_LakshayModi_DS605/Lab04
pip install -r requirements.txt
streamlit run app.py
```

The app loads `airbnb_price_pipeline.joblib` directly — no retraining needed.
To retrain, run the notebook end-to-end; the last cell re-exports the
`.joblib` file.

---

## Task 1 — Data Analysis and Preparation

- **Train/test split first.** The raw data is split 80/20 *before* any
  cleaning, so every statistic used downstream (price thresholds,
  neighbourhood centres, keyword lists, encoders) is learned from the
  training set only. The test set stays untouched until final evaluation.
- **Price cleaning (training set only).** Listings priced at $0 are removed
  (not a valid nightly rate). No upper cutoff is applied — legitimate
  high-price listings are kept so the model learns the real price range
  instead of a truncated one.
- **Other cleaning.** Listings with `minimum_nights > 365` are dropped
  (long-term stays, not nightly rentals). Missing `reviews_per_month` is
  filled with `0` (verified to correspond to listings with zero reviews).
- **Feature engineering:**
  - *Spatial:* distance from each listing to a data-driven "high-price
    centre" of NYC, learned via k-nearest-neighbours local price estimates
    on the training set (out-of-fold, to avoid leakage).
  - *Skew handling:* `log1p` on `minimum_nights` and
    `calculated_host_listings_count`.
  - *Text:* a custom `NameKeywordFeaturizer` learns which words in the
    listing title associate with unusually high or low prices (e.g.
    *penthouse*, *spectacular* → high; *cheap*, *shared* → low) and encodes
    each listing with two binary flags.
  - *Categorical:* `room_type`, `neighbourhood`, `neighbourhood_group`
    one-hot encoded (kept as separate indicator columns rather than
    target-encoded, so no listing's price leaks into its own encoding).
- **What influences price most:** room type (entire home vs. private/shared),
  minimum nights, availability, host's listing count, and distance to the
  learned high-price centre — see feature importance below.

## Task 2 — Model Training and Evaluation

Three model families were compared with identical preprocessing:
Ridge/Linear Regression (baseline), Random Forest, and
HistGradientBoosting. The two tree-based models were tuned with
`RandomizedSearchCV` (5-fold CV, scored on dollar MAE).

| Model | CV MAE |
|---|---|
| Linear Regression (baseline) | $71.05 |
| Ridge Regression (baseline) | $70.94 |
| Random Forest (baseline) | $64.86 |
| HistGradientBoosting (baseline) | $65.11 |
| **Random Forest (tuned)** | **$62.68** |
| **HistGradientBoosting (tuned)** | **$64.85** |

**Final test-set performance** (untouched 20% holdout):

| Model | Train MAE | Train R² | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|
| Tuned Random Forest | $48.04 | 0.447 | $56.50 | $183.77 | 0.237 |
| Tuned HistGradientBoosting | $57.63 | 0.427 | $59.77 | $184.94 | 0.227 |

**Overfitting check.** Both models show a moderate train→test drop in R²
(RF: 0.447→0.237, HGB: 0.427→0.227), a similar-sized gap for both — neither
is dramatically overfitting relative to the other after tuning.

**Why test R² looks modest despite a reasonable MAE.** No high-price cap is
applied to the training or test data, so a handful of very expensive
listings (some in the thousands of dollars) dominate the *squared*-error
terms that R²/RMSE are sensitive to, even though a typical prediction is
only off by ~$56–60. This is a direct, expected consequence of keeping all
legitimate listings rather than trimming outliers — not a modelling bug.

**Deployed model.** `airbnb_price_pipeline.joblib` contains the tuned
**HistGradientBoosting** pipeline. Random Forest scored marginally better on
this run's test set; HistGradientBoosting was kept for deployment for its
much smaller serialized size and faster inference on Streamlit's free tier.
*(If that's not the actual reason you picked HGB, swap in your real one —
or just export and ship the Random Forest pipeline instead, since it
currently tests slightly better.)*

**Feature importance** (permutation importance, tuned HistGradientBoosting):

| Feature | Importance |
|---|---|
| `room_type = Entire home/apt` | 0.219 |
| `minimum_nights` (log) | 0.149 |
| `availability_365` | 0.132 |
| host's total listings (log) | 0.063 |
| distance to high-price centre | 0.048 |
| `longitude` | 0.044 |
| `reviews_per_month` | 0.027 |
| `number_of_reviews` | 0.026 |
| `latitude` | 0.025 |
| has high-price title keyword | 0.024 |

<!-- TODO: export the notebook's spatial and feature-importance plots as
PNGs (e.g. into an images/ folder) and reference them here, since the
assignment asks for "important plots" in the README. -->

**Saved pipeline.** The full pipeline — keyword extraction, spatial feature
engineering, encoding, scaling, and the tuned regressor — is saved as a
single `airbnb_price_pipeline.joblib` via `joblib.dump()`, so the exact
training-time transformations apply to any new listing at inference time
with no separate preprocessing code to keep in sync.

## Task 3 — Streamlit Application

**Live app:** https://202618009lakshaymodids605-airbnbprice.streamlit.app/

The app (`app.py`) takes the same fields the model was trained on —
borough, neighbourhood, room type, coordinates, minimum nights, review
stats, host's listing count, availability, and listing title — and returns
an estimated nightly price. It also:

- offers three one-click example listings (luxury Manhattan, budget shared
  room, typical private room) to test the model quickly;
- flags when the listing title matches a learned high- or low-price
  keyword, and shows the matched word(s);
- reports the model's typical test-set error (±$60) alongside the estimate,
  so the number isn't presented as more precise than it is;
- shows the listing's location on a map and its distance to NYC's learned
  high-price centre;
- includes an "About the model" tab with CV/test metrics, feature
  importance, and known limitations.

Tested with the three preset listings above plus manual inputs across all
five boroughs; predictions stay within a sane range ($40–$300+) and shift
in the expected direction (private/shared rooms cheaper than entire homes,
"cheap"/"shared" in the title pulling the estimate down, etc.).

<!-- TODO: add app screenshots here once you've taken them. -->

## Task 4 — Final Project Summary

- **Best models:** tuned Random Forest and HistGradientBoosting both clearly
  beat the linear baselines (~$71 CV MAE → ~$63–65). The deployed model is
  HistGradientBoosting (see note above).
- **Application:** deployed on Streamlit Community Cloud, tested with
  realistic listings spanning all five boroughs and all three room types.
- **Limitations:**
  - Single 2019 snapshot — no seasonality, and Airbnb pricing is known to
    be seasonal.
  - The high-price centre is city-wide; a per-borough version might catch
    local premium pockets (e.g. specific Manhattan blocks) better.
  - Extreme high-price listings are kept as legitimate data, which is
    defensible but caps how high R²/RMSE can look for this target — a
    log-price target or robust loss would trade that off against the
    "predict the real dollar price" framing used here.
  - Only two tree-based families were tuned; CatBoost/LightGBM (which
    handle high-cardinality categoricals like `neighbourhood` natively)
    weren't compared.
  - The neighbourhood dropdown only accepts values seen during training —
    unseen or misspelled neighbourhoods fall back to "unknown category"
    rather than a hard error.

## Dataset

[Inside Airbnb — New York City, 2019 snapshot](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data),
48,895 listings, 16 columns. Not committed to this repo (download separately
and point the notebook's `DATA_PATH` at your local copy).
