"""
app.py — NYC Airbnb Price Predictor (Streamlit UI)

Thin UI layer on top of model_pipeline.predict_price(). All model
logic (feature engineering, preprocessing, the tuned regressor) lives
in model_pipeline.py / airbnb_price_pipeline.joblib — this file only
collects listing details from the user and displays the result.
"""

import math

import pandas as pd
import streamlit as st

from model_pipeline import predict_price, get_model_insights

# ------------------------------------------------------------------
# Static numbers copied from the notebook's final test-set evaluation
# (Tuned HistGradientBoosting — the model actually shipped in
# airbnb_price_pipeline.joblib). Update these if you retrain/re-export.
# ------------------------------------------------------------------
TEST_MAE = 59.77
TEST_R2 = 0.227
CV_MAE = 64.85

FEATURE_IMPORTANCE = {
    "room_type = Entire home/apt": 0.219,
    "minimum_nights (log)": 0.149,
    "availability_365": 0.132,
    "host's total listings (log)": 0.063,
    "distance to high-price centre": 0.048,
    "longitude": 0.044,
    "reviews_per_month": 0.027,
    "number_of_reviews": 0.026,
    "latitude": 0.025,
    "has high-price title keyword": 0.024,
}

st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_insights():
    return get_model_insights()


def haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8  # Earth radius in miles
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def tokenize(name):
    import re
    words = re.findall(r"[a-z]+", (name or "").lower())
    return {w for w in words if len(w) >= 3}


# ------------------------------------------------------------------
# Load the pipeline once and surface an error banner (instead of a
# stack trace) if the .joblib file is missing next to this script.
# ------------------------------------------------------------------
try:
    insights = load_insights()
    model_ready = True
except FileNotFoundError as e:
    model_ready = False
    insights = None

st.title("🏠 NYC Airbnb Price Predictor")
st.caption(
    "Estimates a nightly listing price from the 2019 Inside Airbnb NYC data, "
    "using the same preprocessing + tuned HistGradientBoosting pipeline built "
    "in the DS605 Lab 4 notebook."
)

if not model_ready:
    st.error(
        "Trained pipeline not found. Make sure `airbnb_price_pipeline.joblib` "
        "sits next to `app.py` and `model_pipeline.py`."
    )
    st.stop()

DEFAULTS = {
    "borough": "Manhattan",
    "neighbourhood": "Midtown",
    "room_type": "Entire home/apt",
    "latitude": 40.7549,
    "longitude": -73.9840,
    "minimum_nights": 2,
    "number_of_reviews": 15,
    "reviews_per_month": 1.2,
    "host_listings": 1,
    "availability_365": 200,
    "name": "Cozy loft near Times Square",
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

PRESETS = {
    "🏙️ Manhattan luxury": {
        "borough": "Manhattan", "neighbourhood": "Tribeca",
        "room_type": "Entire home/apt", "latitude": 40.7163, "longitude": -74.0086,
        "minimum_nights": 3, "number_of_reviews": 8, "reviews_per_month": 0.6,
        "host_listings": 2, "availability_365": 120,
        "name": "Spectacular penthouse loft with skyline views",
    },
    "🛏️ Budget shared room": {
        "borough": "Brooklyn", "neighbourhood": "Bushwick",
        "room_type": "Shared room", "latitude": 40.6958, "longitude": -73.9171,
        "minimum_nights": 1, "number_of_reviews": 42, "reviews_per_month": 3.1,
        "host_listings": 4, "availability_365": 300,
        "name": "Cheap shared room near the subway",
    },
    "🏡 Typical private room": {
        "borough": "Queens", "neighbourhood": "Astoria",
        "room_type": "Private room", "latitude": 40.7643, "longitude": -73.9235,
        "minimum_nights": 2, "number_of_reviews": 20, "reviews_per_month": 1.4,
        "host_listings": 1, "availability_365": 180,
        "name": "Comfortable private room near the park",
    },
}


def apply_preset(preset):
    for k, v in preset.items():
        st.session_state[k] = v


predict_tab, about_tab = st.tabs(["🔮 Predict a price", "📊 About the model"])

with predict_tab:
    st.write("**Try an example, or fill in your own listing below.**")
    preset_cols = st.columns(len(PRESETS))
    for col, (label, preset) in zip(preset_cols, PRESETS.items()):
        col.button(
            label, use_container_width=True,
            on_click=apply_preset, args=(preset,),
        )

    with st.form("listing_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.selectbox(
                "Borough", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
                key="borough",
            )
            st.selectbox("Neighbourhood", insights["neighbourhoods"], key="neighbourhood")
            st.selectbox(
                "Room type", ["Entire home/apt", "Private room", "Shared room"],
                key="room_type",
            )

        with col2:
            st.number_input("Latitude", format="%.5f", key="latitude")
            st.number_input("Longitude", format="%.5f", key="longitude")
            st.number_input(
                "Minimum nights", min_value=1, max_value=365, key="minimum_nights",
                help="Listings requiring over a year's stay are excluded from training.",
            )

        with col3:
            st.number_input("Number of reviews", min_value=0, key="number_of_reviews")
            st.number_input("Reviews per month", min_value=0.0, step=0.1, key="reviews_per_month")
            st.number_input("Host's total listings", min_value=1, key="host_listings")
            st.number_input(
                "Availability (days/yr)", min_value=0, max_value=365, key="availability_365",
            )

        st.text_input(
            "Listing title", key="name",
            help="A few words as they'd appear in the Airbnb listing title.",
        )

        submitted = st.form_submit_button("Predict price", use_container_width=True)

    if submitted:
        listing = {
            "latitude": st.session_state.latitude,
            "longitude": st.session_state.longitude,
            "neighbourhood": st.session_state.neighbourhood,
            "neighbourhood_group": st.session_state.borough,
            "room_type": st.session_state.room_type,
            "minimum_nights": st.session_state.minimum_nights,
            "number_of_reviews": st.session_state.number_of_reviews,
            "reviews_per_month": st.session_state.reviews_per_month,
            "calculated_host_listings_count": st.session_state.host_listings,
            "availability_365": st.session_state.availability_365,
            "name": st.session_state.name,
        }

        try:
            price = predict_price(listing)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
        else:
            centre_lat, centre_lon = insights["high_price_centre"]
            dist_miles = haversine_miles(
                listing["latitude"], listing["longitude"], centre_lat, centre_lon
            )
            tokens = tokenize(listing["name"])
            hit_high = tokens & set(insights["high_price_keywords"])
            hit_low = tokens & set(insights["low_price_keywords"])

            st.divider()
            result_col, map_col = st.columns([1, 1])

            with result_col:
                st.metric("Estimated nightly price", f"${price:,.0f}")
                st.caption(
                    f"On held-out test listings this model is typically off by "
                    f"about **±${TEST_MAE:,.0f}** (mean absolute error), so treat "
                    f"this as a ballpark, not a quote."
                )
                st.metric("Distance to NYC's learned high-price centre", f"{dist_miles:.1f} mi")

                if hit_high:
                    st.success(f"Title matched high-price keyword(s): {', '.join(sorted(hit_high))}")
                if hit_low:
                    st.warning(f"Title matched low-price keyword(s): {', '.join(sorted(hit_low))}")
                if not hit_high and not hit_low:
                    st.caption("Title didn't match any learned high/low-price keywords.")

            with map_col:
                st.map(
                    pd.DataFrame(
                        {"lat": [listing["latitude"]], "lon": [listing["longitude"]]}
                    ),
                    size=60,
                )

with about_tab:
    st.subheader("Model")
    st.write(
        "A tuned **HistGradientBoostingRegressor**, wrapped in a scikit-learn "
        "`Pipeline` that handles keyword extraction from the listing title, "
        "spatial feature engineering (distance to a learned high-price centre), "
        "log-transforms of skewed counts, and one-hot encoding of categoricals — "
        "all fit on training data only, evaluated on an untouched test split. "
        "Full methodology is in the notebook."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Cross-val MAE", f"${CV_MAE:,.2f}")
    m2.metric("Test MAE", f"${TEST_MAE:,.2f}")
    m3.metric("Test R²", f"{TEST_R2:.3f}")

    st.caption(
        "R² looks modest because a handful of very expensive listings dominate "
        "squared-error terms — no high-price cap was applied, by design, so the "
        "test set keeps every legitimate listing including outliers. See the "
        "notebook's conclusion for the full explanation."
    )

    st.subheader("What drives the prediction")
    importance_df = pd.DataFrame(
        {"feature": list(FEATURE_IMPORTANCE.keys()), "importance": list(FEATURE_IMPORTANCE.values())}
    ).set_index("feature")
    st.bar_chart(importance_df, horizontal=True)
    st.caption(
        "Permutation importance for the deployed model, computed once in the "
        "training notebook — not recalculated live."
    )

    st.subheader("Limitations")
    st.markdown(
        "- Trained on a single 2019 snapshot — no seasonality signal.\n"
        "- The high-price centre is city-wide, not per-borough.\n"
        "- Extreme high-price listings are kept as legitimate data, which caps "
        "how high test R² can realistically get.\n"
        "- Predicting an unfamiliar neighbourhood, or a listing very unlike "
        "anything in the training data, will be less reliable."
    )
