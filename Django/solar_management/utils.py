import os
import pickle
import numpy as np
import pandas as pd
from django.conf import settings
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
from datetime import datetime, timedelta

def get_weather_forecast(lat=12.8911, lon=80.0815):
    """
    Fetches weather forecast data using the OpenWeatherMap One Call API
    and provides advice based on rain prediction.
    
    :param lat: Latitude of the location (default: Vandalur, Chennai)
    :param lon: Longitude of the location (default: Vandalur, Chennai)
    :return: Advice string based on the weather forecast
    """

    # Get the API key from Django settings
    api_key = settings.OPENWEATHER_API_KEY
    
    # OpenWeatherMap API endpoint for One Call API
    url = f"http://api.weatherapi.com/v1/forecast.json?key=04c7bcd904174973b92182555240909&q=Chennai&days=11&aqi=no&alerts=no"
    
    try:
        # Make a request to the OpenWeatherMap API
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses

        # Parse the JSON response
        weather_data = response.json()
        return weather_data
        # rain = weather_data['forecast']['forecastday'][0]['day']['totalprecip_mm']

        # # Check if rain is forecasted in the next day's weather
        # if rain > 2.5:
        #     return f"Reduce power usage by 30% to prepare for reduced input tomorrow, rain mm: {rain}"
        # else:
        #     return f"No need to save power, rain mm: {rain}"
    
    
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"


def check_if_panel_defective(panel_data):
    # Simulate defect detection
    if panel_data['input_power'] <= 0:
        return True
    return False

# utils.py

def calculate_mppt(voltage, current, prev_voltage, prev_current):
    """
    Calculates the Maximum Power Point (MPPT) using the Perturb and Observe method.
    
    :param voltage: The current voltage of the solar panel.
    :param current: The current output of the solar panel.
    :param prev_voltage: The previous voltage of the solar panel.
    :param prev_current: The previous current output of the solar panel.
    :return: The updated voltage and power at the Maximum Power Point.
    """
    
    # Calculate the power at the current and previous points
    power = voltage * current
    prev_power = prev_voltage * prev_current
    
    # Check how the power has changed
    if power > prev_power:
        # If power has increased, continue in the same direction
        next_voltage = voltage + 0.1  # Increment voltage slightly
    else:
        # If power has decreased, reverse the direction
        next_voltage = voltage - 0.1  # Decrement voltage slightly

    # Update current for the new voltage
    # In a real scenario, you would retrieve the actual current for this voltage
    next_current = current  # Placeholder for actual current
    
    return next_voltage, power

# solar_management/utils.py

class GenerationModelHandler:
    _instance = None
    _model = None
    _x_scaler = None
    _y_scaler = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self._load_model()
    
    def _load_model(self):
        """Load the TensorFlow generation model and its scalers"""
        try:
            # Define paths to model and scalers
            model_path = os.path.join(settings.BASE_DIR, 'generation_model.h5')
            x_scaler_path = os.path.join(settings.BASE_DIR, 'generation_X_scaler.pkl')
            y_scaler_path = os.path.join(settings.BASE_DIR, 'generation_y_scaler.pkl')
            
            print(f"Attempting to load model from: {model_path}")
            print(f"X scaler path: {x_scaler_path}")
            print(f"Y scaler path: {y_scaler_path}")
            
            # Check if files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            if not os.path.exists(x_scaler_path):
                raise FileNotFoundError(f"X scaler file not found at {x_scaler_path}")
            if not os.path.exists(y_scaler_path):
                raise FileNotFoundError(f"Y scaler file not found at {y_scaler_path}")
            
            # Load the model without compiling it first
            self._model = load_model(model_path, compile=False)
            
            # Recompile the model with the necessary metrics
            self._model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            # Load the scalers
            with open(x_scaler_path, 'rb') as f:
                self._x_scaler = pickle.load(f)
            
            with open(y_scaler_path, 'rb') as f:
                self._y_scaler = pickle.load(f)
                
            print("Generation model and scalers loaded successfully")
        except Exception as e:
            print(f"Error loading generation model: {str(e)}")
            import traceback
            traceback.print_exc()
            self._model = None
            self._x_scaler = None
            self._y_scaler = None
    
    def predict(self, input_data):
        """
        Make predictions with the generation model
        
        Args:
            input_data: Numpy array or DataFrame with features
                       [temperature, cloud_cover, wind_speed, solar_irradiance, 
                        hour, day_of_year, month]
        
        Returns:
            Array of predictions (generation_kw)
        """
        if self._model is None or self._x_scaler is None or self._y_scaler is None:
            self._load_model()
            if self._model is None:
                raise ValueError("Generation model could not be loaded")
        
        # Convert to numpy array if DataFrame
        if isinstance(input_data, pd.DataFrame):
            features = ['temperature', 'cloud_cover', 'wind_speed', 'solar_irradiance', 
                        'hour', 'day_of_year', 'month']
            input_data = input_data[features].values
        
        # Scale input data
        input_scaled = self._x_scaler.transform(input_data)
        
        # Make predictions
        pred_scaled = self._model.predict(input_scaled)
        
        # Inverse transform to get actual values
        predictions = self._y_scaler.inverse_transform(pred_scaled)
        
        return predictions

def generate_sample_solar_data(days=1, panels_capacity_kw=10):
    """Generate sample solar data for testing the generation model"""
    # Create date range with hourly intervals
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    date_range = [start_date + timedelta(hours=i) for i in range(days*24)]
    
    # Initialize dataframe
    df = pd.DataFrame({
        'datetime': date_range,
        'day_of_year': [d.timetuple().tm_yday for d in date_range],
        'hour': [d.hour for d in date_range],
        'month': [d.month for d in date_range]
    })
    
    # Add season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
    df['season'] = df['month'].apply(lambda m: (m%12+3)//3)
    
    # Temperature (°C)
    df['temperature'] = df.apply(lambda row: 
                       15 + 10 * np.sin(2 * np.pi * (row['day_of_year']-172)/365) +  
                       5 * np.sin(2 * np.pi * row['hour']/24 - np.pi/2) + 
                       np.random.normal(0, 2),
                       axis=1)
    
    # Cloud cover (0-1)
    cloud_base = np.random.beta(2, 5, len(df))
    season_cloud_factor = {1: 1.4, 2: 1.0, 3: 0.7, 4: 1.1}
    df['cloud_cover'] = df.apply(
        lambda row: min(1, cloud_base[row.name] * season_cloud_factor[row['season']]),
        axis=1
    )
    
    # Wind speed (m/s)
    df['wind_speed'] = df.apply(
        lambda row: abs(np.random.normal(
            4 + 2 * (row['season'] == 1 or row['season'] == 4), 2)),
        axis=1
    )
    
    # Solar irradiance
    df['solar_irradiance_base'] = df.apply(
        lambda row: max(0, 1000 * np.sin(np.pi * row['hour']/12) 
                      if 6 <= row['hour'] <= 18 else 0),
        axis=1
    )
    
    day_length_factor = df['day_of_year'].apply(
        lambda d: 0.7 + 0.6 * np.sin(2 * np.pi * (d - 172) / 365)
    )
    df['solar_irradiance'] = df['solar_irradiance_base'] * day_length_factor * (1 - 0.75 * df['cloud_cover'])
    
    # Drop intermediate columns
    df = df.drop(['solar_irradiance_base', 'season'], axis=1, errors='ignore')
    
    return df
