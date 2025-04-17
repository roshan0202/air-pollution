import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Set page configuration
st.set_page_config(
    page_title="Air Quality Index Prediction",
    page_icon="🌬️",
    layout="wide"
)

def main():
    # Add a title and description
    st.title("Air Quality Index (AQI) Prediction App")
    st.markdown("""
    This application predicts Air Quality Index based on various pollutant levels.
    Upload your data or use the interactive inputs to make predictions.
    """)
    
    # Create sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Training", "Make Predictions"])
    
    # Initialize session state for model and data
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if page == "Home":
        show_home_page()
    elif page == "Data Exploration":
        show_data_exploration()
    elif page == "Model Training":
        show_model_training()
    elif page == "Make Predictions":
        show_prediction_page()

def show_home_page():
    st.header("Welcome to the Air Quality Index Prediction App")
    
    st.subheader("About this application")
    st.write("""
    This application uses machine learning to predict Air Quality Index (AQI) based on various air pollutant concentrations.
    The model uses the following features:
    - PM2.5: Fine particulate matter
    - PM10: Coarse particulate matter  
    - NO2: Nitrogen Dioxide
    - SO2: Sulfur Dioxide
    - CO: Carbon Monoxide
    - O3: Ozone
    
    To get started, upload your air quality data or use the built-in sample data for exploration.
    """)
    
    st.subheader("How to use this app")
    st.markdown("""
    1. **Data Exploration**: Upload and visualize your air quality data
    2. **Model Training**: Train a Random Forest model on your data
    3. **Make Predictions**: Input pollutant values to predict AQI
    """)

def load_data():
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        return data
    else:
        st.info("Please upload a CSV file or use the 'Load Sample Data' option.")
        if st.button("Load Sample Data"):
            try:
                # Try to load the original data file
                data = pd.read_csv("city_day.csv")
                st.session_state.data = data
                return data
            except FileNotFoundError:
                # Generate synthetic data if file not found
                st.warning("Sample file not found. Using synthetic data instead.")
                np.random.seed(42)
                n_samples = 1000
                data = pd.DataFrame({
                    'PM2.5': np.random.uniform(0, 250, n_samples),
                    'PM10': np.random.uniform(0, 350, n_samples),
                    'NO2': np.random.uniform(0, 100, n_samples),
                    'SO2': np.random.uniform(0, 40, n_samples),
                    'CO': np.random.uniform(0, 5, n_samples),
                    'O3': np.random.uniform(0, 180, n_samples),
                    'AQI': np.random.uniform(0, 500, n_samples)
                })
                st.session_state.data = data
                return data
        return None

def show_data_exploration():
    st.header("Data Exploration")
    
    data = load_data()
    
    if data is not None:
        st.subheader("Data Preview")
        st.dataframe(data.head())
        
        st.subheader("Data Summary")
        st.write(data.describe())
        
        st.subheader("Check Missing Values")
        missing_data = data.isnull().sum()
        st.write(missing_data)
        
        # Data visualization
        st.subheader("Data Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']
            available_features = [f for f in features if f in data.columns]
            
            if all(feature in data.columns for feature in available_features):
                corr = data[available_features].corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Some required columns are missing for correlation analysis.")
        
        with col2:
            st.subheader("Feature Distributions")
            feature_to_plot = st.selectbox("Select feature to visualize:", available_features)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data[feature_to_plot].dropna(), kde=True, ax=ax)
            st.pyplot(fig)
        
        st.subheader("Feature vs. AQI")
        if 'AQI' in data.columns:
            feature_to_scatter = st.selectbox("Select feature to plot against AQI:", 
                                            [f for f in available_features if f != 'AQI'])
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=data[feature_to_scatter], y=data['AQI'], alpha=0.6, ax=ax)
            plt.title(f"{feature_to_scatter} vs AQI")
            st.pyplot(fig)
        else:
            st.warning("AQI column not found in the dataset.")

def show_model_training():
    st.header("Model Training")
    
    data = st.session_state.data if st.session_state.data is not None else load_data()
    
    if data is not None:
        st.write("Training a Random Forest Regressor model on your data...")
        
        # Define features and target
        features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        target = 'AQI'
        
        # Check if all features and target are in the dataset
        missing_columns = [col for col in features + [target] if col not in data.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return
        
        # Extract features and target
        X = data[features]
        y = data[target]
        
        # Handle missing values
        X.fillna(X.mean(), inplace=True)
        y.fillna(y.mean(), inplace=True)
        
        # Model parameters
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("Number of trees (n_estimators)", 10, 200, 100)
            max_depth = st.slider("Maximum depth of trees (max_depth)", 5, 30, 20)
        
        with col2:
            min_samples_split = st.slider("Minimum samples to split (min_samples_split)", 2, 10, 2)
            min_samples_leaf = st.slider("Minimum samples in leaf (min_samples_leaf)", 1, 10, 1)
        
        test_size = st.slider("Test size (percentage)", 0.1, 0.5, 0.2)
        
        if st.button("Train Model"):
            with st.spinner("Training in progress..."):
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Train model
                rf_model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )
                
                rf_model.fit(X_train, y_train)
                st.session_state.model = rf_model
                
                # Evaluate model
                y_pred = rf_model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                st.success("Model training completed!")
                st.write(f"Model performance - MAE: {mae:.2f}, R²: {r2:.2f}")
                
                # Feature importance
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
                plt.title("Feature Importance")
                st.pyplot(fig)
                
                # Save model option
                if st.button("Save Model"):
                    joblib.dump(rf_model, "best_rf_model.pkl")
                    st.success("Model saved as 'best_rf_model.pkl'")
        
        # Option to load a pre-trained model
        st.subheader("Or Load Pre-trained Model")
        uploaded_model = st.file_uploader("Upload a trained model (.pkl file)", type=["pkl"])
        if uploaded_model is not None:
            try:
                model = joblib.load(uploaded_model)
                st.session_state.model = model
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")

def show_prediction_page():
    st.header("Make AQI Predictions")
    
    # Check if model exists
    if st.session_state.model is None:
        st.warning("No trained model found. Please train a model first or upload a pre-trained model.")
        if st.button("Load Default Model"):
            try:
                model = joblib.load("best_rf_model.pkl")
                st.session_state.model = model
                st.success("Default model loaded successfully!")
            except FileNotFoundError:
                st.error("Default model file not found. Please train a new model.")
                return
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return
    
    # Input form for single prediction
    st.subheader("Single Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=50.0)
        no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, value=20.0)
    
    with col2:
        pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=100.0)
        so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, value=10.0)
    
    with col3:
        co = st.number_input("CO (mg/m³)", min_value=0.0, value=1.0)
        o3 = st.number_input("O3 (µg/m³)", min_value=0.0, value=40.0)
    
    if st.button("Predict AQI"):
        # Create input array for prediction
        input_data = pd.DataFrame({
            'PM2.5': [pm25],
            'PM10': [pm10],
            'NO2': [no2],
            'SO2': [so2],
            'CO': [co],
            'O3': [o3]
        })
        
        # Make prediction
        prediction = st.session_state.model.predict(input_data)[0]
        
        # Determine AQI category
        category, color = get_aqi_category(prediction)
        
        # Display prediction with styling
        st.subheader("Prediction Result")
        st.markdown(f"""
        <div style="background-color:{color}; padding:10px; border-radius:5px;">
            <h3 style="color:white;">Predicted AQI: {prediction:.2f}</h3>
            <h4 style="color:white;">Category: {category}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Add interpretation
        st.subheader("Interpretation")
        st.write(get_aqi_interpretation(category))
    
    # Batch prediction
    st.subheader("Batch Prediction")
    st.write("Upload a CSV file with pollutant data to make batch predictions.")
    
    batch_file = st.file_uploader("Upload CSV with features", type=["csv"], key="batch_upload")
    
    if batch_file is not None:
        try:
            batch_data = pd.read_csv(batch_file)
            features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
            
            # Check if all required features are present
            missing_features = [f for f in features if f not in batch_data.columns]
            if missing_features:
                st.error(f"Missing required features in upload: {missing_features}")
            else:
                # Handle missing values
                batch_data[features].fillna(batch_data[features].mean(), inplace=True)
                
                # Make predictions
                batch_predictions = st.session_state.model.predict(batch_data[features])
                
                # Add predictions to dataframe
                result_df = batch_data.copy()
                result_df['Predicted AQI'] = batch_predictions
                
                # Display results
                st.subheader("Batch Prediction Results")
                st.dataframe(result_df)
                
                # Download option
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name="aqi_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing batch file: {e}")

def get_aqi_category(aqi):
    """Return AQI category and color based on AQI value"""
    if aqi <= 50:
        return "Good", "#00e400"
    elif aqi <= 100:
        return "Moderate", "#ffff00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi <= 200:
        return "Unhealthy", "#ff0000"
    elif aqi <= 300:
        return "Very Unhealthy", "#99004c"
    else:
        return "Hazardous", "#7e0023"

def get_aqi_interpretation(category):
    """Return health implications based on AQI category"""
    interpretations = {
        "Good": "Air quality is considered satisfactory, and air pollution poses little or no risk.",
        "Moderate": "Air quality is acceptable; however, for some pollutants there may be a moderate health concern for a very small number of people.",
        "Unhealthy for Sensitive Groups": "Members of sensitive groups may experience health effects. The general public is not likely to be affected.",
        "Unhealthy": "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.",
        "Very Unhealthy": "Health warnings of emergency conditions. The entire population is more likely to be affected.",
        "Hazardous": "Health alert: everyone may experience more serious health effects."
    }
    return interpretations.get(category, "No interpretation available.")

if __name__ == "__main__":
    main()
