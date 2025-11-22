import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------
# Load and preprocess dataset
# ---------------------------------------------------
df = pd.read_csv("Clean_Dataset.csv")

cols_to_drop = ["flight", "flight_code", "num_code", "Unnamed: 0"]
for col in cols_to_drop:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

label_cols = ["airline", "source_city", "departure_time", "stops",
              "arrival_time", "destination_city", "class"]

encoders = {}
for col in label_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder

X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42
)
model.fit(X_train, y_train)


# ---------------------------------------------------
# Predict single flight price
# ---------------------------------------------------
def predict_price(details: dict):
    sample = pd.DataFrame([details])
    for col in label_cols:
        sample[col] = encoders[col].transform(sample[col])
    return model.predict(sample)[0]


# ---------------------------------------------------
# Recommend best day to book for next 30 days
# ---------------------------------------------------
def best_day_to_book(base_details: dict):

    predicted_prices = []

    for d in range(30):
        temp = base_details.copy()
        temp["days_left"] = d
        predicted_prices.append(predict_price(temp))

    # risk-adjusted decision scoring
    scores = []
    for i in range(30):
        today = predicted_prices[i]
        future = predicted_prices[i + 1:] if i < 29 else [today]
        avg_future = np.mean(future)
        volatility = np.mean(np.abs(np.diff(future))) if len(future) > 1 else 0

        score = (today * 0.6) + (avg_future * 0.3) + (volatility * 0.1)
        scores.append(score)

    best_index = int(np.argmin(scores))

    return best_index, predicted_prices
