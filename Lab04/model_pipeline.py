"""
model_pipeline.py

Feature engineering + preprocessing + inference for the DS605 Lab 4
Airbnb price prediction pipeline. Loads the fitted pipeline saved as
`airbnb_price_pipeline.joblib` (preprocessing + tuned
HistGradientBoostingRegressor) and exposes predict_price() for the
Streamlit app.

The two transformer classes below must match the notebook's
definitions exactly (same fields/logic) — the saved pipeline was
pickled with its class references pointed at this module.
"""

import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors


# ============================================================
# Feature columns expected by the pipeline
# ============================================================

FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "neighbourhood",
    "neighbourhood_group",
    "room_type",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "name",
]

PIPELINE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "airbnb_price_pipeline.joblib",
)


# ============================================================
# NameKeywordFeaturizer — exact copy from the notebook
# ============================================================

class NameKeywordFeaturizer(BaseEstimator, TransformerMixin):
    """
    Learns which words in the listing `name` are associated with
    unusually high or unusually low price, using TRAINING data only,
    then flags each listing with two binary indicators:

        has_high_price_keyword
        has_low_price_keyword
    """

    _STOPWORDS = {
        "the", "a", "an", "and", "or", "in", "on", "at", "by", "for",
        "of", "to", "with", "near", "is", "are", "this", "your", "you",
        "min", "mins", "new", "bed", "beds", "bedroom", "bedrooms",
        "room", "rooms", "apt", "apartment", "nyc",
    }

    def __init__(self, top_k=15, min_count=30):
        self.top_k = top_k
        self.min_count = min_count

    @staticmethod
    def _tokenize(name):
        if not isinstance(name, str):
            return []
        words = re.findall(r"[a-z]+", name.lower())
        return [w for w in words if len(w) >= 3]

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("NameKeywordFeaturizer requires y during fit.")

        y = pd.Series(y).reset_index(drop=True)
        names = X["name"].reset_index(drop=True)

        global_mean_price = y.mean()

        word_price_sum = Counter()
        word_count = Counter()

        for name, price in zip(names, y):
            tokens = set(self._tokenize(name)) - self._STOPWORDS
            for word in tokens:
                word_price_sum[word] += price
                word_count[word] += 1

        word_stats = []
        for word, count in word_count.items():
            if count < self.min_count:
                continue
            mean_price = word_price_sum[word] / count
            lift = mean_price / global_mean_price
            word_stats.append((word, count, mean_price, lift))

        word_stats_df = pd.DataFrame(
            word_stats, columns=["word", "count", "mean_price", "lift"]
        )

        high = word_stats_df.sort_values("lift", ascending=False).head(self.top_k)
        low = word_stats_df.sort_values("lift", ascending=True).head(self.top_k)

        self.high_price_keywords_ = set(high["word"])
        self.low_price_keywords_ = set(low["word"])
        self.keyword_stats_ = word_stats_df

        return self

    def transform(self, X):
        X = X.copy()
        tokens = X["name"].apply(self._tokenize)

        X["has_high_price_keyword"] = tokens.apply(
            lambda ws: int(any(w in self.high_price_keywords_ for w in ws))
        )
        X["has_low_price_keyword"] = tokens.apply(
            lambda ws: int(any(w in self.low_price_keywords_ for w in ws))
        )
        return X


# ============================================================
# AirbnbFeatureEngineer — exact copy from the notebook
# ============================================================

class AirbnbFeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self, k_neighbours=20, city_price_percentile=99,
                 n_folds=5, random_state=42):
        self.k_neighbours = k_neighbours
        self.city_price_percentile = city_price_percentile
        self.n_folds = n_folds
        self.random_state = random_state

    def _learn_high_price_centre(self, X, y):
        coords = X[["latitude", "longitude"]].to_numpy()
        n_neighbours = min(self.k_neighbours + 1, len(coords))

        knn = NearestNeighbors(n_neighbors=n_neighbours)
        knn.fit(coords)
        distances, indices = knn.kneighbors(coords)
        neighbour_indices = indices[:, 1:]

        local_mean_price = np.mean(y.to_numpy()[neighbour_indices], axis=1)
        threshold = np.percentile(local_mean_price, self.city_price_percentile)
        high_price_mask = local_mean_price >= threshold
        high_price_coords = coords[high_price_mask]

        self.high_price_centre_ = (
            high_price_coords[:, 0].mean(),
            high_price_coords[:, 1].mean(),
        )

    def fit(self, X, y=None):
        X = X.copy()
        if y is None:
            raise ValueError("AirbnbFeatureEngineer requires y during fit.")
        self._learn_high_price_centre(X, y)
        return self

    def transform(self, X):
        X = X.copy()
        high_lat, high_lon = self.high_price_centre_

        X["dist_to_high_price_center"] = np.sqrt(
            (X["latitude"] - high_lat) ** 2 + (X["longitude"] - high_lon) ** 2
        )
        X["minimum_nights_log"] = np.log1p(X["minimum_nights"])
        X["host_listings_log"] = np.log1p(X["calculated_host_listings_count"])
        X["is_never_available"] = (X["availability_365"] == 0).astype(int)

        return X

    def fit_transform(self, X, y=None, **fit_params):
        X = X.copy()
        self.fit(X, y)
        X_transformed = self.transform(X)

        oof_distance = pd.Series(index=X.index, dtype=float)
        kfold = KFold(n_splits=self.n_folds, shuffle=True,
                       random_state=self.random_state)

        for train_idx, valid_idx in kfold.split(X):
            X_fold_train = X.iloc[train_idx]
            X_fold_valid = X.iloc[valid_idx]
            y_fold_train = y.iloc[train_idx]

            fold_engineer = AirbnbFeatureEngineer(
                k_neighbours=self.k_neighbours,
                city_price_percentile=self.city_price_percentile,
                n_folds=self.n_folds,
                random_state=self.random_state,
            )
            fold_engineer.fit(X_fold_train, y_fold_train)
            valid_transformed = fold_engineer.transform(X_fold_valid)

            oof_distance.loc[X_fold_valid.index] = valid_transformed[
                "dist_to_high_price_center"
            ]

        X_transformed["dist_to_high_price_center"] = oof_distance
        return X_transformed


# ============================================================
# Pipeline loading (cached) + public inference function
# ============================================================

_pipeline = None


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(PIPELINE_PATH):
            raise FileNotFoundError(
                f"Trained pipeline not found at {PIPELINE_PATH}. "
                "Run the training/export step in the notebook first "
                "(joblib.dump(final_pipeline, 'airbnb_price_pipeline.joblib'))."
            )
        _pipeline = joblib.load(PIPELINE_PATH)
    return _pipeline


def predict_price(input_data):
    """
    Predict nightly price for one or more Airbnb listings.

    input_data : dict (single listing) or list[dict] / DataFrame
                 (multiple listings). Must contain FEATURE_COLUMNS:
                 latitude, longitude, neighbourhood, neighbourhood_group,
                 room_type, minimum_nights, number_of_reviews,
                 reviews_per_month, calculated_host_listings_count,
                 availability_365, name

    Returns: float (single listing) or np.ndarray (multiple listings)
    """
    pipeline = _load_pipeline()

    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
        single = True
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
        single = False
    else:
        df = pd.DataFrame(input_data)
        single = False

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[FEATURE_COLUMNS]

    preds = pipeline.predict(df)
    preds = np.maximum(preds, 0)

    return float(preds[0]) if single else preds


if __name__ == "__main__":
    # Quick smoke test — run `python model_pipeline.py` after the
    # joblib file exists to sanity-check a single prediction.
    sample = {
        "latitude": 40.7484,
        "longitude": -73.9857,
        "neighbourhood": "Midtown",
        "neighbourhood_group": "Manhattan",
        "room_type": "Entire home/apt",
        "minimum_nights": 2,
        "number_of_reviews": 15,
        "reviews_per_month": 1.2,
        "calculated_host_listings_count": 1,
        "availability_365": 200,
        "name": "Cozy loft near Times Square",
    }
    print("Predicted price:", predict_price(sample))
