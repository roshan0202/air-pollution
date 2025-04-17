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
    df = pd.read_csv('city_day.csv')  # Ensure the dataset is in the same directory
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

# Explaining about the model and pollutants
st.write('### About the Model')
st.write("""
The **AQI Predictor** uses a **Random Forest Regressor model** to estimate the air quality index (AQI) based on the levels of four key air pollutants: 
- **CO (Carbon Monoxide)**: A colorless and odorless gas that can harm human health when present in high concentrations.
- **NO2 (Nitrogen Dioxide)**: A toxic gas that contributes to the formation of ground-level ozone and fine particulate matter.
- **PM2.5 (Particulate Matter 2.5)**: Tiny particles in the air that can be inhaled into the lungs, leading to various respiratory issues.
- **PM10 (Particulate Matter 10)**: Larger particles, but still harmful when inhaled, causing respiratory distress and other health concerns.

This model was trained on historical data collected from various cities and uses these pollutants to estimate how safe or unsafe the air quality is for daily activities. By inputting the levels of these pollutants, the model predicts an AQI value, which helps users understand the air quality and make informed decisions regarding their health and safety.

### How the Model Works
1. **Data**: The model was trained using pollutant levels (CO, NO2, PM2.5, PM10) and their corresponding AQI values.
2. **Prediction**: By entering the pollutant levels, the model processes the data and estimates the AQI, classifying it into one of the categories: Good, Satisfactory, Moderate, Poor, Very Poor, or Severe.
3. **Usage**: This tool is useful for anyone needing quick insights into the air quality of their surroundings based on real-time pollutant readings.
""")

st.write('### Input Pollutant Levels')
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

    st.success(f'🏙️ Predicted AQI: {prediction:.2f} - {category}')

    # How the model works section after prediction
    st.write('### How the Model Works After Prediction')
    st.write("""
    1. **Data**: The model was trained using pollutant levels (CO, NO2, PM2.5, PM10) and their corresponding AQI values.
    2. **Prediction**: By entering the pollutant levels, the model processes the data and estimates the AQI, classifying it into one of the categories: Good, Satisfactory, Moderate, Poor, Very Poor, or Severe.
    3. **Usage**: This tool is useful for anyone needing quick insights into the air quality of their surroundings based on real-time pollutant readings.
    """)

# Optional: Model Evaluation (if needed)
with st.expander("Show Model Evaluation Metrics 📈"):
    y_pred = model.predict(X_test_scaled)
    st.write("Mean Absolute Error (MAE):", round(mean_absolute_error(y_test, y_pred), 2))
    st.write("R² Score:", round(r2_score(y_test, y_pred), 2))
