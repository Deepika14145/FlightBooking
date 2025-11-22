# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

# ---------- Config: dataset path (uses your uploaded file) ----------
DATA_PATH = "Clean_Dataset.csv"

# ---------- Load and basic cleaning ----------
df = pd.read_csv(DATA_PATH)

# Drop columns that are identifiers or not useful for prediction
for c in ["flight", "flight_code", "num_code", "Unnamed: 0"]:
    if c in df.columns:
        df.drop(columns=[c], inplace=True)

# Ensure columns exist and consistent names
expected_cols = ["airline", "source_city", "departure_time", "stops",
                 "arrival_time", "destination_city", "class",
                 "duration", "days_left", "price"]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing expected columns in dataset: {missing}")

# ---------- Build encoders and option lists ----------
label_cols = ["airline", "source_city", "departure_time", "stops",
              "arrival_time", "destination_city", "class"]

encoders = {}
mappings = {}           # mapping dicts for fast safe encoding
model_options = {}      # option lists to populate UI selectboxes

for col in label_cols:
    le = LabelEncoder()
    # Replace NaNs with a placeholder
    df[col] = df[col].fillna("UNKNOWN")
    le.fit(df[col].astype(str).values)
    encoders[col] = le
    # mapping: raw string -> integer encoding
    mappings[col] = {v: int(k) for k, v in enumerate(le.classes_)}
    model_options[col] = list(le.classes_)

# ---------- Prepare features and target ----------
X = df.drop(columns=["price"])
y = df["price"].astype(float)

# For safety, ensure numeric cols are numeric
X["duration"] = pd.to_numeric(X["duration"], errors="coerce").fillna(X["duration"].median())
X["days_left"] = pd.to_numeric(X["days_left"], errors="coerce").fillna(X["days_left"].median())

# Replace categorical string columns with their encoded integers for training
X_encoded = X.copy()
for col in label_cols:
    X_encoded[col] = X_encoded[col].map(mappings[col]).fillna(-1).astype(int)

# ---------- Train/test split and model training ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42
)

model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Optional: quick validation metric (printed when module imported)
try:
    preds = model.predict(X_test)
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test, preds)
    print(f"[model.py] Trained RandomForest — MAE on holdout: {mae:.2f}")
except Exception:
    pass


# ---------- Helper: safe encode for a single sample dict ----------
def _safe_encode_sample(sample: dict):
    """
    Accepts a dict with keys matching feature names and returns
    a DataFrame with encoded categorical columns and numeric columns.
    Unknown categories map to -1.
    """
    s = sample.copy()
    # fill missing categorical with "UNKNOWN"
    for col in label_cols:
        if col not in s or s[col] is None:
            s[col] = "UNKNOWN"

    # Convert to DataFrame
    df_s = pd.DataFrame([s], columns=X.columns)

    # Ensure numeric columns are numeric
    df_s["duration"] = pd.to_numeric(df_s["duration"], errors="coerce").fillna(X["duration"].median())
    df_s["days_left"] = pd.to_numeric(df_s["days_left"], errors="coerce").fillna(X["days_left"].median())

    # Encode categorical columns using mappings; unseen -> -1
    for col in label_cols:
        val = str(df_s.at[0, col])
        df_s[col] = mappings[col].get(val, -1)

    # ensure integer dtype for encoded cols
    for col in label_cols:
        df_s[col] = df_s[col].astype(int)

    return df_s


# ---------- Public function: predict price for single flight details ----------
def predict_price(details: dict) -> float:
    """
    details: dict with keys:
      'airline','source_city','departure_time','stops',
      'arrival_time','destination_city','class','duration','days_left'
    returns: predicted price (float)
    """
    sample_df = _safe_encode_sample(details)
    pred = model.predict(sample_df)[0]
    return float(pred)


# ---------- Public function: recommend best day to book in next 30 days ----------
def best_day_to_book(base_details: dict, horizon_days: int = 30):
    """
    Simulates booking on each day from 0..horizon_days-1 (0 = book today).
    Interpretation:
      - base_details['days_left'] is days until travel if booked today.
      - If booking on day i (i days from today), the model sees days_left = max(0, base_days_left - i).
    Returns:
      best_index: integer (0..horizon_days-1) recommended day to book (days from today)
      predicted_prices: list of predicted prices length horizon_days
    """
    if "days_left" not in base_details:
        raise ValueError("base_details must include 'days_left' (days until travel if booked today).")

    base_days_left = int(base_details.get("days_left", 0))
    predicted_prices = []

    for i in range(horizon_days):
        # days left at travel if booking i days from now
        days_left_if_booked_i = max(0, base_days_left - i)
        temp = base_details.copy()
        temp["days_left"] = int(days_left_if_booked_i)
        predicted_prices.append(predict_price(temp))

    # Risk-adjusted scoring: consider today's price, avg of future, and volatility
    scores = []
    for i in range(horizon_days):
        today = predicted_prices[i]
        future = predicted_prices[i + 1:] if i < (horizon_days - 1) else [today]
        avg_future = np.mean(future) if len(future) > 0 else today
        volatility = float(np.mean(np.abs(np.diff(future)))) if len(future) > 1 else 0.0

        score = (today * 0.6) + (avg_future * 0.3) + (volatility * 0.1)
        scores.append(score)

    best_index = int(np.argmin(scores))
    return best_index, predicted_prices
