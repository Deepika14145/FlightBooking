# app.py
import streamlit as st
import pandas as pd
from model import predict_price, best_day_to_book, model_options

st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️", layout="wide")

st.title("✈️ Flight Price & Best Booking Day (Instant Prediction)")
st.write("Fill in flight details. The app will predict today's price and also forecast the best day to book within the next 30 days.")

# Helper to fetch valid dataset options
def opt(key, default=[]):
    return model_options.get(key, default)

# ---------------------- Input Section ----------------------
st.sidebar.header("Flight Details")

airline = st.sidebar.selectbox("Airline", opt("airline"))
source_city = st.sidebar.selectbox("Source City", opt("source_city"))
destination_city = st.sidebar.selectbox("Destination City", opt("destination_city"))
departure_time = st.sidebar.selectbox("Departure Time", opt("departure_time"))
arrival_time = st.sidebar.selectbox("Arrival Time", opt("arrival_time"))
stops = st.sidebar.selectbox("Stops", opt("stops"))
travel_class = st.sidebar.selectbox("Class", opt("class"))

duration = st.sidebar.number_input("Duration (hours)", 0.5, 48.0, 2.0, step=0.1)
days_left = st.sidebar.number_input("Days until Travel (book today)", 0, 365, 10, step=1)
horizon = st.sidebar.slider("Prediction Range (Days)", 7, 60, 30)

details = {
    "airline": airline,
    "source_city": source_city,
    "departure_time": departure_time,
    "stops": stops,
    "arrival_time": arrival_time,
    "destination_city": destination_city,
    "class": travel_class,
    "duration": float(duration),
    "days_left": int(days_left)
}

st.markdown("---")
st.subheader("📊 Prediction Results")

# ---------------------- Main Prediction Trigger ----------------------
if st.button("Predict Price & Best Booking Day"):
    
    # Predict price today
    today_price = int(predict_price(details))

    # Predict best day
    best_day, price_list = best_day_to_book(details, horizon_days=horizon)
    best_price = int(min(price_list))

    # Two-column metric display
    c1, c2 = st.columns(2)
    c1.metric("Price if Booked Today", f"₹{today_price:,}")
    c2.metric("Best Price in Forecast", f"₹{best_price:,}", delta=f"Best Day: {best_day}")

    st.markdown("---")

    # ---------------------- Line Chart ----------------------
    df_chart = pd.DataFrame({
        "Day": list(range(horizon)),
        "Predicted Price": [int(p) for p in price_list]
    }).set_index("Day")

    st.subheader("📈 Price Forecast for Next 30 Days")
    st.line_chart(df_chart)

    # ---------------------- Table View ----------------------
    st.subheader("📄 Detailed Price List")
    st.dataframe(df_chart)

else:
    st.info("Click the button to get full prediction, forecast and charts.")

st.markdown("---")
st.caption("Model trained on `/mnt/data/Clean_Dataset.csv`. Unknown categories are safely handled.")
