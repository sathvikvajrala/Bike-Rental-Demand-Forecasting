import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Load trained model
model = pickle.load(open("bike_demand_model.pkl", "rb"))

st.title("🚲 Bike Rental Demand Prediction")

# ---------- USER INPUTS ----------
season = st.selectbox("Season", ["spring", "summer", "fall", "winter"])
yr = st.selectbox("Year", [0, 1])
mnth = st.slider("Month", 1, 12, 6)
hr = st.slider("Hour", 0, 23, 10)
holiday = st.selectbox("Holiday", [0, 1])
weekday = st.selectbox("Weekday (0=Sun, 6=Sat)", [0,1,2,3,4,5,6])
workingday = st.selectbox("Working Day", [0, 1])
weather = st.selectbox("Weather", ["Clear", "Mist", "Light Snow", "Heavy Rain"])

temp = st.slider("Temperature", 0.0, 1.0, 0.6)
atemp = st.slider("Feels Like Temp", 0.0, 1.0, 0.62)
hum = st.slider("Humidity", 0.0, 1.0, 0.55)
windspeed = st.slider("Windspeed", 0.0, 1.0, 0.18)

# ---------- FEATURE ENGINEERING ----------
day = datetime.now().day
quarter = (mnth - 1) // 3 + 1
is_weekend = 1 if weekday in [5, 6] else 0

temp_sq = temp ** 2
hum_temp_interaction = hum * temp
windspeed_log = np.log1p(windspeed)

is_peak_hour = 1 if hr in [7, 8, 9, 17, 18, 19] else 0
is_high_season = 1 if season in ["summer", "fall"] else 0
comfort_index = (1 - abs(temp - 0.6)) * (1 - hum)
is_weekend_or_holiday = 1 if (is_weekend or holiday) else 0
bad_weather = 1 if weather in ["Heavy Rain", "Light Snow"] else 0
timestamp_ns = int(pd.Timestamp.now().value)

FEATURES = [
    'yr','mnth','hr','temp','atemp','hum','windspeed',
    'day','quarter','is_weekend','temp_sq',
    'hum_temp_interaction','windspeed_log',
    'season_fall','season_spring','season_summer','season_winter',
    'weathersit_Clear','weathersit_Heavy Rain',
    'weathersit_Light Snow','weathersit_Mist',
    'holiday_No','holiday_Yes',
    'weekday_0','weekday_1','weekday_2','weekday_3',
    'weekday_4','weekday_5','weekday_6',
    'workingday_No work','workingday_Working Day',
    'is_peak_hour','is_high_season','comfort_index',
    'is_weekend_or_holiday','bad_weather','timestamp_ns'
]

feature_dict = {f: 0 for f in FEATURES}

feature_dict.update({
    'yr': yr,
    'mnth': mnth,
    'hr': hr,
    'temp': temp,
    'atemp': atemp,
    'hum': hum,
    'windspeed': windspeed,
    'day': day,
    'quarter': quarter,
    'is_weekend': is_weekend,
    'temp_sq': temp_sq,
    'hum_temp_interaction': hum_temp_interaction,
    'windspeed_log': windspeed_log,
    'is_peak_hour': is_peak_hour,
    'is_high_season': is_high_season,
    'comfort_index': comfort_index,
    'is_weekend_or_holiday': is_weekend_or_holiday,
    'bad_weather': bad_weather,
    'timestamp_ns': timestamp_ns
})

feature_dict[f'season_{season}'] = 1
feature_dict[f'weathersit_{weather}'] = 1
feature_dict[f'weekday_{weekday}'] = 1
feature_dict['holiday_Yes' if holiday else 'holiday_No'] = 1
feature_dict['workingday_Working Day' if workingday else 'workingday_No work'] = 1

input_df = pd.DataFrame([feature_dict])[FEATURES]

# ---------- PREDICTION ----------
if st.button("Predict Bike Demand"):
    prediction = model.predict(input_df)[0]
    st.success(f"🚴 Predicted Bike Demand: {int(prediction)}")
