# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# FIRST Streamlit command
st.set_page_config(page_title="AQI Predictor 🌎", page_icon="🌿", layout="wide")

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

# Add an introductory explanation about AQI and its impact
st.write("""
    The Air Quality Index (AQI) is a measure of how clean or polluted the air is and how it impacts your health. 
    The higher the AQI, the greater the level of air pollution and the greater the health concern. This tool uses 
    the concentrations of four key pollutants (CO, NO2, PM2.5, and PM10) to predict the AQI and classify it into 
    one of the following categories:
    
    - **Good**: 0 - 50
    - **Satisfactory**: 51 - 100
    - **Moderate**: 101 - 150
    - **Poor**: 151 - 200
    - **Very Poor**: 201 - 300
    - **Severe**: 301 and above
""")

# User Input
st.write("Enter the pollutant levels below to predict AQI:")

# Pollutant Input Fields
CO = st.number_input('CO (mg/m³)', min_value=0.0, value=1.5, help="Carbon Monoxide (CO) level in mg/m³")
NO2 = st.number_input('NO2 (µg/m³)', min_value=0.0, value=25.3, help="Nitrogen Dioxide (NO2) level in µg/m³")
PM25 = st.number_input('PM2.5 (µg/m³)', min_value=0.0, value=82.5, help="PM2.5 (Particulate Matter) level in µg/m³")
PM10 = st.number_input('PM10 (µg/m³)', min_value=0.0, value=122.0, help="PM10 (Particulate Matter) level in µg/m³")

# Prediction Button
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

    st.success(f'🏙️ **Predicted AQI**: {prediction:.2f} - **{category}**')

# Optional: Model Evaluation (if needed)
with st.expander("Show Model Evaluation Metrics 📈"):
    y_pred = model.predict(X_test_scaled)
    st.write("Mean Absolute Error (MAE):", round(mean_absolute_error(y_test, y_pred), 2))
    st.write("R² Score:", round(r2_score(y_test, y_pred), 2))

# Visualizations for Understanding the Model
st.write("### Feature Importance")
st.write("""
    Below are the feature importances based on the Random Forest model. 
    It shows how much each pollutant contributes to the AQI prediction.
""")
feature_importances = model.feature_importances_
features_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plotting Feature Importance
fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=features_df, ax=ax, palette='viridis')
st.pyplot(fig)

# Visualizing the distribution of AQI
st.write("### AQI Distribution in the Dataset")
fig, ax = plt.subplots()
sns.histplot(df[target], kde=True, ax=ax, color='skyblue', bins=30)
ax.set_title("AQI Distribution")
ax.set_xlabel("AQI")
ax.set_ylabel("Frequency")
st.pyplot(fig)
