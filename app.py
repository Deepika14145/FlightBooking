import streamlit as st
import pandas as pd
import plotly.express as px
from model import predict_price, best_day_to_book

# Page config
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Flight Price Prediction & Best Booking Day")
st.markdown("Enter your flight details to get prediction and booking advice.")

# --------------------------
# User Input Form
# --------------------------

airline = st.selectbox("Airline", ["Indigo", "Air India", "SpiceJet", "GoAir", "Vistara", "AirAsia"])
source_city = st.selectbox("Source City", ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Kolkata", "Chennai"])
destination_city = st.selectbox("Destination City", ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Kolkata", "Chennai"])

departure_time = st.selectbox("Departure Time", ["Early Morning", "Morning", "Afternoon", "Evening", "Night", "Late Night"])
arrival_time = st.selectbox("Arrival Time", ["Early Morning", "Morning", "Afternoon", "Evening", "Night", "Late Night"])

stops = st.selectbox("Stops", ["zero", "one", "two_or_more"])
travel_class = st.selectbox("Class", ["Economy", "Business"])

duration = st.number_input("Flight Duration (Hours)", min_value=1.0, max_value=15.0, step=0.1)
days_left = st.number_input("Days Left Before Travel", min_value=0, max_value=60, step=1)

# Build input dictionary
details = {
    "airline": airline,
    "source_city": source_city,
    "departure_time": departure_time,
    "stops": stops,
    "arrival_time": arrival_time,
    "destination_city": destination_city,
    "class": travel_class,
    "duration": duration,
    "days_left": days_left
}

# --------------------------
# Predict Button
# --------------------------
if st.button("Predict Price"):
    price = predict_price(details)
    st.success(f"Estimated Price: ₹{int(price)}")


# --------------------------
# Best Day Recommendation
# --------------------------
if st.button("Recommend Best Booking Day"):

    best_day, prices = best_day_to_book(details)

    st.info(f"Best Day to Book: **Day {best_day}** from today")

    df_plot = pd.DataFrame({
        "Day": list(range(30)),
        "Price": prices
    })

    fig = px.line(df_plot, x="Day", y="Price",
                  title="30-Day Predicted Price Trend",
                  markers=True)
    st.plotly_chart(fig)
