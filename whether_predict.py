import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go

# Set page config for better appearance
st.set_page_config(
    page_title="Weather Predictor",
    page_icon="W",
    layout="wide"
)

# Enhanced CSS for better color contrast and interactivity
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1a2a6c, #b21f1f, #1a2a6c);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        padding: 20px;
        border-radius: 15px;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: linear-gradient(135deg, #2c3e50, #4a6491);
        background-size: 400% 400%;
        animation: gradientApp 10s ease infinite;
    }
    
    @keyframes gradientApp {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .title {
        text-align: center;
        color: #ffffff;
        font-size: 3rem;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.5), 0 0 20px rgba(255, 255, 255, 0.3);
        margin-bottom: 2rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .prediction-card {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: 20px 0;
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        background: rgba(255, 255, 255, 0.2);
    }
    
    .season-display {
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        animation: fadeIn 1s ease-in;
        transition: all 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .winter { 
        background: linear-gradient(135deg, #3498db, #2980b9);
        box-shadow: 0 0 20px rgba(52, 152, 219, 0.5);
    }
    
    .spring { 
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        box-shadow: 0 0 20px rgba(46, 204, 113, 0.5);
    }
    
    .summer { 
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        box-shadow: 0 0 20px rgba(231, 76, 60, 0.5);
    }
    
    .autumn { 
        background: linear-gradient(135deg, #e67e22, #d35400);
        box-shadow: 0 0 20px rgba(230, 126, 34, 0.5);
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: 20px 0;
        transition: all 0.3s ease;
    }
    
    .chart-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* Button enhancement */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 15px 30px;
        font-size: 1.2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        background: rgba(255, 255, 255, 0.15);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        color: white;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Text adjustments for better readability */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    
    p, div, span {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Footer styling */
    footer {
        color: rgba(255, 255, 255, 0.7) !important;
    }
</style>
""", unsafe_allow_html=True)

class WeatherPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def fetch_historical_weather_data(self):
        dates = pd.date_range(start='2020-01-01', end='2024-06-14')
        n = len(dates)
        data = {
            'date': dates,
            'temperature': np.random.normal(20, 5, n),
            'humidity': np.random.normal(60, 10, n),
            'pressure': np.random.normal(1013, 10, n),
            'wind_speed': np.random.normal(10, 3, n)
        }
        return pd.DataFrame(data)
    
    def prepare_data(self, df):
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        
        features = ['month', 'day', 'humidity', 'pressure', 'wind_speed']
        target = 'temperature'
        
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
    
    def predict_weather(self, input_features):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        input_scaled = self.scaler.transform([input_features])
        prediction = self.model.predict(input_scaled)
        
        return prediction[0]

@st.cache_resource(show_spinner=True)
def load_predictor():
    predictor = WeatherPredictor()
    data = predictor.fetch_historical_weather_data()
    X_train, X_test, y_train, y_test = predictor.prepare_data(data)
    predictor.train_model(X_train, y_train)
    return predictor, data

def month_to_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Autumn"
    else:
        return "Unknown"

# Animated title
st.markdown("<h1 class='title'>Dynamic Weather Temperature Predictor</h1>", unsafe_allow_html=True)

# Load trained model and full dataset for visualization
predictor, historical_data = load_predictor()

# Create tabs for better organization
tab1, tab2 = st.tabs(["Prediction", "Historical Data"])

with tab1:
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    st.subheader("Input Weather Conditions")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        month = st.slider("Month", 1, 12, 6, help="Select the month for prediction")
        day = st.slider("Day", 1, 31, 15, help="Select the day for prediction")
        humidity = st.slider("Humidity (%)", 0, 100, 65, help="Relative humidity percentage")
    
    with col2:
        pressure = st.slider("Pressure (hPa)", 900, 1100, 1010, help="Atmospheric pressure in hPa")
        wind_speed = st.slider("Wind Speed (km/h)", 0, 50, 12, help="Wind speed in kilometers per hour")
    
    input_features = [month, day, humidity, pressure, wind_speed]
    season = month_to_season(month)
    
    # Display season with color coding
    season_class = season.lower()
    st.markdown(f"<div class='season-display {season_class}'>Current Season: {season}</div>", unsafe_allow_html=True)
    
    # Animated prediction button
    if st.button("Predict Temperature", use_container_width=True):
        try:
            predicted_temp = predictor.predict_weather(input_features)
            
            # Display result in an animated card
            st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
            st.subheader("Prediction Results")
            
            # Temperature display with color coding
            if predicted_temp < 0:
                temp_color = "#3498db"  # Blue for freezing
                weather_desc = "Freezing Cold"
            elif predicted_temp < 10:
                temp_color = "#3498db"  # Blue for cold
                weather_desc = "Cold"
            elif predicted_temp < 20:
                temp_color = "#2ecc71"  # Green for mild
                weather_desc = "Mild"
            elif predicted_temp < 30:
                temp_color = "#f39c12"  # Orange for warm
                weather_desc = "Warm"
            else:
                temp_color = "#e74c3c"  # Red for hot
                weather_desc = "Hot"
            
            st.markdown(f"<h2 style='color: {temp_color}; text-align: center; font-size: 3rem; text-shadow: 0 0 10px rgba(255,255,255,0.5);'>{predicted_temp:.2f} °C</h2>", unsafe_allow_html=True)
            
            # Additional weather info display
            st.markdown(f"<p style='text-align: center; font-size: 1.5rem; margin-top: 10px;'>Season: {season}</p>", unsafe_allow_html=True)
            
            st.markdown(f"<p style='text-align: center; font-size: 1.5rem; margin-top: 10px;'>Feels like: {weather_desc}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error in prediction: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.subheader("Historical Temperature Trends (2020-2024)")
    
    # Interactive Plotly chart
    fig = px.line(historical_data, x='date', y='temperature', 
                  title='Temperature Over Time',
                  labels={'temperature': 'Temperature (°C)', 'date': 'Date'},
                  line_shape='spline')
    fig.update_traces(line=dict(color='#ff6b6b', width=4))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(size=24, color='white'),
        xaxis=dict(color='rgba(255,255,255,0.8)'),
        yaxis=dict(color='rgba(255,255,255,0.8)')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional statistics
    st.subheader("Weather Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Temp", f"{historical_data['temperature'].mean():.1f}°C")
    with col2:
        st.metric("Max Temp", f"{historical_data['temperature'].max():.1f}°C")
    with col3:
        st.metric("Min Temp", f"{historical_data['temperature'].min():.1f}°C")
    with col4:
        st.metric("Humidity Avg", f"{historical_data['humidity'].mean():.1f}%")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: rgba(255, 255, 255, 0.7); font-size: 1.1rem;'>Dynamic Weather Predictor - Powered by Machine Learning</p>", unsafe_allow_html=True)