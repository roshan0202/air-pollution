# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# FIRST Streamlit command
st.set_page_config(page_title="AQI Predictor 🌎", page_icon="🌿")

# Step 1: Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('city_day.csv')
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    return df

df = load_data()

# Step 2: Prepare features and target (only CO, NO2, PM2.5, PM10 for prediction)
features = ['CO', 'NO2', 'PM2.5', 'PM10']
target = 'AQI'
X = df[features]
y = df[target]

# Step 3: Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 5: Streamlit UI
st.title('🌎 Air Quality Index (AQI) Predictor')
st.write('Enter the pollutant levels below to predict AQI:')

# User Input (now only CO, NO2, PM2.5, and PM10)
CO = st.number_input('CO (mg/m³)', min_value=0.0, value=1.5)
NO2 = st.number_input('NO2 (µg/m³)', min_value=0.0, value=25.3)
PM25 = st.number_input('PM2.5 (µg/m³)', min_value=0.0, value=82.5)
PM10 = st.number_input('PM10 (µg/m³)', min_value=0.0, value=122.0)

if st.button('Predict AQI'):
    input_data = np.array([[CO, NO2, PM25, PM10]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success(f'🏙️ Predicted AQI: {prediction:.2f}')

# Optional: Model Evaluation (if needed)
with st.expander("Show Model Evaluation Metrics 📈"):
    y_pred = model.predict(X_test_scaled)
    st.write("Mean Absolute Error (MAE):", round(mean_absolute_error(y_test, y_pred), 2))
    st.write("R² Score:", round(r2_score(y_test, y_pred), 2))
