import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Set page configuration
st.set_page_config(
    page_title="AQI Prediction App",
    page_icon="🌍",
    layout="wide"
)

# Load and train model (only done once when app starts)
@st.cache_resource
def load_and_train_model():
    # Either include your dataset directly or load from URL
    # For example, you could upload the dataset to GitHub and load it via raw URL:
    # dataset_url = "https://raw.githubusercontent.com/yourusername/yourrepo/main/city_day.csv"
    # df = pd.read_csv(dataset_url)
    
    # For demonstration, let's use a simplified approach with sample data
    try:
        df = pd.read_csv("city_day.csv")  # Try to load local file
    except:
        # If file not found, create sample training data
        # This is just for demonstration - replace with your actual data loading logic
        st.warning("Dataset not found - using sample data instead. For production, include your actual dataset.")
        df = pd.DataFrame({
            'PM2.5': np.random.uniform(0, 200, 1000),
            'PM10': np.random.uniform(0, 300, 1000),
            'NO2': np.random.uniform(0, 100, 1000),
            'SO2': np.random.uniform(0, 50, 1000),
            'CO': np.random.uniform(0, 10, 1000),
            'O3': np.random.uniform(0, 150, 1000),
            'AQI': np.random.uniform(0, 500, 1000)
        })
    
    # Define features and target
    features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    target = 'AQI'
    
    # Extract features and target
    X = df[features]
    y = df[target]
    
    # Handle missing values
    X.fillna(X.mean(), inplace=True)
    y.fillna(y.mean(), inplace=True)
    
    # Train the model with optimized hyperparameters
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X, y)
    
    return model, features

# Function to make predictions
def predict_aqi(model, input_data):
    prediction = model.predict(input_data)[0]
    return prediction

# Function to determine AQI category and color
def get_aqi_category(aqi_value):
    if aqi_value <= 50:
        return "Good", "#00e400"
    elif aqi_value <= 100:
        return "Moderate", "#ffff00"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi_value <= 200:
        return "Unhealthy", "#ff0000"
    elif aqi_value <= 300:
        return "Very Unhealthy", "#99004c"
    else:
        return "Hazardous", "#7e0023"

# Main function
def main():
    # Load or train model
    model, features = load_and_train_model()
    
    # Rest of your app code remains the same...
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["AQI Predictor", "About AQI", "Data Insights"])
    
    # AQI Predictor Page
    if page == "AQI Predictor":
        st.title("Air Quality Index (AQI) Predictor")
        st.write("Enter pollutant concentrations to predict the Air Quality Index")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pm25 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, max_value=1000.0, value=35.0, step=1.0)
            pm10 = st.number_input("PM10 (μg/m³)", min_value=0.0, max_value=1000.0, value=50.0, step=1.0)
            no2 = st.number_input("NO2 (μg/m³)", min_value=0.0, max_value=500.0, value=30.0, step=1.0)
        
        with col2:
            so2 = st.number_input("SO2 (μg/m³)", min_value=0.0, max_value=500.0, value=20.0, step=1.0)
            co = st.number_input("CO (mg/m³)", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
            o3 = st.number_input("O3 (μg/m³)", min_value=0.0, max_value=500.0, value=40.0, step=1.0)
        
        if st.button("Predict AQI"):
            # Create input dataframe for prediction
            input_df = pd.DataFrame({
                'PM2.5': [pm25],
                'PM10': [pm10],
                'NO2': [no2],
                'SO2': [so2],
                'CO': [co],
                'O3': [o3]
            })
            
            # Make prediction
            prediction = predict_aqi(model, input_df)
            
            # Get AQI category and color
            category, color = get_aqi_category(prediction)
            
            # Display results
            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style="background-color:{color};padding:20px;border-radius:10px;text-align:center;">
                    <h1 style="color:black;">{prediction:.1f}</h1>
                    <h3 style="color:black;">{category}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Health implications
                st.subheader("Health Implications")
                if prediction <= 50:
                    st.write("Air quality is considered satisfactory, and air pollution poses little or no risk.")
                elif prediction <= 100:
                    st.write("Air quality is acceptable; however, there may be a moderate health concern for a very small number of people.")
                elif prediction <= 150:
                    st.write("Members of sensitive groups may experience health effects. The general public is not likely to be affected.")
                elif prediction <= 200:
                    st.write("Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.")
                elif prediction <= 300:
                    st.write("Health warnings of emergency conditions. The entire population is more likely to be affected.")
                else:
                    st.write("Health alert: everyone may experience more serious health effects.")

    # About AQI Page content...
    # Data Insights Page content...
    # [The rest of your code remains the same]

# Run the app
if __name__ == "__main__":
    main()
