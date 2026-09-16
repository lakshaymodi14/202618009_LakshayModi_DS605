import streamlit as st
from model_pipeline import predict_price

st.set_page_config(page_title="NYC Airbnb Price Predictor", page_icon="🏠")
st.title("🏠 NYC Airbnb Price Predictor")

with st.form("listing_form"):
    col1, col2 = st.columns(2)

    with col1:
        neighbourhood_group = st.selectbox(
            "Borough",
            ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
        )
        neighbourhood = st.text_input("Neighbourhood", "Midtown")
        room_type = st.selectbox(
            "Room type",
            ["Entire home/apt", "Private room", "Shared room"],
        )
        latitude = st.number_input("Latitude", value=40.7484, format="%.5f")
        longitude = st.number_input("Longitude", value=-73.9857, format="%.5f")

    with col2:
        minimum_nights = st.number_input("Minimum nights", min_value=1, value=2)
        number_of_reviews = st.number_input("Number of reviews", min_value=0, value=15)
        reviews_per_month = st.number_input("Reviews per month", min_value=0.0, value=1.2)
        calculated_host_listings_count = st.number_input(
            "Host's total listings", min_value=1, value=1
        )
        availability_365 = st.number_input(
            "Availability (days/yr)", min_value=0, max_value=365, value=200
        )

    name = st.text_input("Listing title", "Cozy loft near Times Square")

    submitted = st.form_submit_button("Predict price")

if submitted:
    listing = {
        "latitude": latitude,
        "longitude": longitude,
        "neighbourhood": neighbourhood,
        "neighbourhood_group": neighbourhood_group,
        "room_type": room_type,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": calculated_host_listings_count,
        "availability_365": availability_365,
        "name": name,
    }

    try:
        price = predict_price(listing)
        st.success(f"Estimated nightly price: **${price:,.2f}**")
    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
