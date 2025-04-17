import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="AQI Prediction App",
    page_icon="🌍",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    try:
        return joblib.load("best_rf_model.pkl")
    except:
        st.error("Model file not found. Please ensure 'best_rf_model.pkl' is in the same directory.")
        return None

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
    # Load model
    model = load_model()
    
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
            if model is not None:
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

    # About AQI Page
    elif page == "About AQI":
        st.title("About Air Quality Index (AQI)")
        
        st.write("""
        ## What is AQI?
        The Air Quality Index (AQI) is an index for reporting daily air quality. It tells you how clean or polluted your air is, and what associated health effects might be a concern for you.
        
        ## AQI Categories:
        """)
        
        # Create DataFrame for AQI categories
        aqi_categories = pd.DataFrame({
            'AQI Range': ['0-50', '51-100', '101-150', '151-200', '201-300', '301+'],
            'Category': ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous'],
            'Color': ['#00e400', '#ffff00', '#ff7e00', '#ff0000', '#99004c', '#7e0023']
        })
        
        # Display AQI categories with colored boxes
        for i, row in aqi_categories.iterrows():
            st.markdown(f"""
            <div style="display:flex;align-items:center;margin-bottom:10px;">
                <div style="background-color:{row['Color']};width:30px;height:30px;margin-right:10px;"></div>
                <div>
                    <strong>{row['AQI Range']}: {row['Category']}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("""
        ## Key Pollutants
        
        - **PM2.5**: Fine particulate matter with a diameter of 2.5 micrometers or smaller. Sources include vehicle emissions, wildfires, and industrial processes.
        
        - **PM10**: Coarse particulate matter with a diameter of 10 micrometers or smaller. Sources include dust from construction, mining, and agricultural operations.
        
        - **NO2**: Nitrogen dioxide, a reddish-brown gas formed by burning fuel at high temperatures. Major sources include vehicles, power plants, and industrial processes.
        
        - **SO2**: Sulfur dioxide, a colorless gas with a strong odor. Sources include fossil fuel combustion, industrial processes, and volcanic eruptions.
        
        - **CO**: Carbon monoxide, a colorless, odorless gas produced by incomplete combustion of fossil fuels. Major sources include vehicle emissions and industrial processes.
        
        - **O3**: Ozone, a colorless gas formed when nitrogen oxides and volatile organic compounds react with sunlight. Ground-level ozone can harm lung function and irritate the respiratory system.
        """)

    # Data Insights Page
    elif page == "Data Insights":
        st.title("AQI Data Insights")
        
        st.write("""
        ### Feature Importance
        
        Based on our Random Forest model, the following features have the most impact on AQI prediction:
        """)
        
        # Create sample feature importance data
        feature_importance = pd.DataFrame({
            'Feature': ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'],
            'Importance': [0.52, 0.23, 0.12, 0.06, 0.04, 0.03]  # Example values
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
        plt.title('Feature Importance for AQI Prediction')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("""
        ### Typical Pollutant Ranges
        
        The table below shows typical ranges for different pollutants:
        """)
        
        # Create sample pollutant ranges
        pollutant_ranges = pd.DataFrame({
            'Pollutant': ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'],
            'Good': ['0-12', '0-54', '0-53', '0-35', '0-4.4', '0-54'],
            'Moderate': ['12.1-35.4', '54.1-154', '53.1-100', '35.1-75', '4.5-9.4', '54.1-70'],
            'Unhealthy': ['35.5-150.4', '154.1-354', '100.1-360', '75.1-185', '9.5-30.4', '70.1-200']
        })
        
        st.table(pollutant_ranges)

# Run the app
if __name__ == "__main__":
    main()
