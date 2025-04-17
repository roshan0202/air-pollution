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
st.write('This app predicts the Air Quality Index (AQI) based on the levels of key pollutants: CO, NO2, PM2.5, and PM10.')
st.write('AQI Categories:')
st.write('1. **Good**: 0 - 50')
st.write('2. **Satisfactory**: 51 - 100')
st.write('3. **Moderate**: 101 - 150')
st.write('4. **Poor**: 151 - 200')
st.write('5. **Very Poor**: 201 - 300')
st.write('6. **Severe**: 301 and above')

# User Input (now only CO, NO2, PM2.5, and PM10)
CO = st.number_input('CO (mg/m³)', min_value=0.0, value=1.5, help="Carbon Monoxide level in mg/m³")
NO2 = st.number_input('NO2 (µg/m³)', min_value=0.0, value=25.3, help="Nitrogen Dioxide level in µg/m³")
PM25 = st.number_input('PM2.5 (µg/m³)', min_value=0.0, value=82.5, help="PM2.5 level in µg/m³")
PM10 = st.number_input('PM10 (µg/m³)', min_value=0.0, value=122.0, help="PM10 level in µg/m³")

if st.button('Predict AQI'):
    # Predict AQI using the model
    input_data = np.array([[CO, NO2, PM25, PM10]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # Classifying AQI into categories
    if prediction <= 50:
        category = "Good"
    elif prediction <= 100:
        category = "Satisfactory"
    elif prediction <= 150:
        category = "Moderate"
    elif prediction <= 200:
        category = "Poor"
    elif prediction <= 300:
        category = "Very Poor"
    else:
        category = "Severe"

    # Displaying the result
    st.success(f'🏙️ Predicted AQI: {prediction:.2f} - {category}')

# Model Evaluation (shown in a collapsible section)
with st.expander("Show Model Evaluation Metrics 📈"):
    y_pred = model.predict(X_test_scaled)
    st.write("Mean Absolute Error (MAE):", round(mean_absolute_error(y_test, y_pred), 2))
    st.write("R² Score:", round(r2_score(y_test, y_pred), 2))

# Step 6: Display additional info about the model
st.markdown("""
### About the Model
This prediction model uses a **Random Forest Regressor** to estimate the AQI based on the levels of common pollutants like **CO**, **NO2**, **PM2.5**, and **PM10**. 

- **CO (Carbon Monoxide)** is a colorless, odorless gas that is harmful when inhaled in large amounts.
- **NO2 (Nitrogen Dioxide)** is a gas that contributes to air pollution and respiratory problems.
- **PM2.5 and PM10** are tiny particles in the air that can affect health when inhaled.

This app helps predict how safe or unsafe the air is based on these pollutant levels.
""")
