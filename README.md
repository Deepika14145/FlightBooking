# Flight Price Prediction using Streamlit

This project predicts flight ticket prices based on airline details, travel timings, stops, duration, and days left for the journey.
The app uses **Streamlit** for the UI and a separate backend module for model loading and prediction.

---

## 📌 Features

* Clean and interactive Streamlit UI
* Dropdowns and number inputs for all flight parameters
* Backend separated from UI logic
* Real-time prediction output
* Extensible for new models or additional parameters

---

## 🗂 Project Structure

```
project/
│
├── app.py                 # Streamlit UI
├── model.py             # Model loading and prediction logic
├── model.pkl              # Trained ML model
├── README.md              # Documentation
└── solution.ipynb
|___ Clean_Dataset.csv    # dataset (obtained from kaggle) 
```

---

## 🧰 Tech Stack

* Python
* Streamlit
* Scikit-learn
* Pandas
* Pickle

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install streamlit pandas scikit-learn
```

### 2. Run the app

```bash
streamlit run app.py
```

---

## 📁 Dataset Format

Example rows from dataset:

```
airline,flight,source_city,departure_time,stops,arrival_time,destination_city,class,duration,days_left,price
SpiceJet,SG-8709,Delhi,Evening,zero,Night,Mumbai,Economy,2.17,1,5953
SpiceJet,SG-8157,Delhi,Early_Morning,zero,Morning,Mumbai,Economy,2.33,1,5953
AirAsia,I5-764,Delhi,Early_Morning,zero,Early_Morning,Mumbai,Economy,2.17,1,5956. etc
```

---

## 📸 Screenshots

*Add your screenshots below:*

```
[ Place Screenshot 1 here ]
<img width="1883" height="946" alt="image" src="https://github.com/user-attachments/assets/6bb32d82-4a6f-43b0-91ad-68b49afa81e6" />
<img width="1644" height="470" alt="image" src="https://github.com/user-attachments/assets/ec57c039-d79e-4837-b3fd-ba07abf0f28f" />

[ Place Screenshot 2 here ]
```

---

## 🛠 Future Improvements

* Add charts for historical price trends
* Include multi-city or return-trip prediction
* Add login and user dashboard
* Deploy on Streamlit Cloud or Render

---

## 🤝 Contribution

Feel free to open issues or improve the UI and model.

---
