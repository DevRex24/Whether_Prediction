# Weather Temperature Predictor

A machine learning-based web application that predicts temperature based on weather conditions using historical data and interactive visualizations.

## Overview

This project is a dynamic weather temperature prediction system built with Streamlit and machine learning. It uses Random Forest Regression to predict temperature based on various weather parameters like humidity, atmospheric pressure, wind speed, and temporal factors.

## Features

- **Temperature Prediction**: Predict temperature based on weather conditions
- **Interactive UI**: Streamlit-based web interface with animated elements
- **Historical Data Visualization**: Interactive charts showing temperature trends
- **Season Detection**: Automatic season classification based on month
- **Real-time Feedback**: Dynamic weather descriptions based on predicted temperature
- **Responsive Design**: Custom CSS styling with gradient backgrounds and animations

## Technology Stack

- **Python 3.x**
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive plotting library

## Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install streamlit numpy pandas scikit-learn plotly
   ```

## Usage

1. Navigate to the project directory
2. Run the Streamlit application:
   ```bash
   streamlit run whether_predict.py
   ```
3. Open your web browser and go to the displayed local URL (typically http://localhost:8501)

## Application Structure

### Main Components

- **WeatherPredictor Class**: Core machine learning model handling
- **Data Generation**: Simulated historical weather data from 2020-2024
- **Feature Engineering**: Month, day, humidity, pressure, and wind speed features
- **Model Training**: Random Forest Regressor with 100 estimators
- **Prediction Interface**: Interactive sliders for input parameters

### Interface Tabs

1. **Prediction Tab**: 
   - Input controls for weather parameters
   - Real-time temperature prediction
   - Season detection and weather classification

2. **Historical Data Tab**:
   - Time series visualization of temperature trends
   - Statistical summary metrics
   - Interactive Plotly charts

## Model Details

### Features Used
- Month (1-12)
- Day (1-31) 
- Humidity (0-100%)
- Atmospheric Pressure (900-1100 hPa)
- Wind Speed (0-50 km/h)

### Target Variable
- Temperature (Celsius)

### Algorithm
- Random Forest Regressor
- 100 decision trees
- StandardScaler for feature normalization
- 80/20 train-test split

## Weather Classifications

The application categorizes predicted temperatures into:

- **Freezing Cold**: Below 0°C
- **Cold**: 0-10°C  
- **Mild**: 10-20°C
- **Warm**: 20-30°C
- **Hot**: Above 30°C

## Season Detection

Automatic season classification based on month:

- **Winter**: December, January, February
- **Spring**: March, April, May
- **Summer**: June, July, August  
- **Autumn**: September, October, November

## Customization

### Styling
The application includes custom CSS for:
- Gradient backgrounds with animations
- Glass-morphism effect cards
- Interactive hover effects
- Responsive design elements
- Color-coded temperature displays

### Data Source
Currently uses simulated data. Can be modified to integrate with:
- Weather APIs (OpenWeatherMap, WeatherAPI, etc.)
- Historical weather databases
- Real-time sensor data

## File Structure

```
whether predict/
├── whether_predict.py    # Main application file
└── README.md            # Project documentation
```

## Performance Metrics

The model uses standard regression metrics for evaluation:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared Score

## Browser Compatibility

Tested and optimized for:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Future Enhancements

Potential improvements and features:
- Integration with real weather APIs
- Extended forecast predictions
- Additional weather parameters (precipitation, UV index)
- Model comparison dashboard
- Export prediction results
- Mobile-responsive optimizations
- User preference settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the application thoroughly
5. Submit a pull request with detailed description

## License

This project is open source and available under standard open source licenses.

## Support

For issues, questions, or suggestions:
- Check existing documentation
- Review the code comments
- Test with different input combinations
- Verify all dependencies are installed correctly

## Technical Notes

- The application generates synthetic historical data for demonstration
- Model training occurs on application startup
- Predictions are cached for performance
- CSS animations may impact performance on slower devices
- Recommended Python version: 3.7+

## Deployment

For production deployment:
- Use cloud platforms (Heroku, Streamlit Cloud, AWS)
- Configure environment variables
- Set up proper logging and monitoring
- Implement data validation and error handling
- Consider containerization with Docker