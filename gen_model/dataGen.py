import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_solar_data(days=365, panels_capacity_kw=10):
    # Create date range with hourly intervals
    start_date = datetime(2024, 1, 1)
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
    
    # Generate weather data
    # Temperature (°C)
    df['temp_raw'] = df.apply(lambda row: 
                           15 + 10 * np.sin(2 * np.pi * (row['day_of_year']-172)/365) +  # Yearly cycle
                           5 * np.sin(2 * np.pi * row['hour']/24 - np.pi/2),             # Daily cycle
                           axis=1)
    
    # Add random variation to temperature
    df['temperature'] = df['temp_raw'] + np.random.normal(0, 2, len(df))
    
    # Cloud cover (0-1)
    df['cloud_cover_base'] = np.random.beta(2, 5, len(df))  # Beta distribution for cloud cover
    
    # Make cloud cover correlated with season (more clouds in winter)
    season_cloud_factor = {1: 1.4, 2: 1.0, 3: 0.7, 4: 1.1}
    df['cloud_cover'] = df.apply(
        lambda row: min(1, row['cloud_cover_base'] * season_cloud_factor[row['season']]),
        axis=1
    )
    
    # Precipitation probability based on cloud cover
    df['precipitation'] = df['cloud_cover'].apply(
        lambda c: max(0, np.random.normal(c*5 - 3, 1)) if c > 0.65 else 0
    )
    
    # Wind speed (m/s) - higher in winter
    df['wind_speed'] = df.apply(
        lambda row: abs(np.random.normal(
            4 + 2 * (row['season'] == 1 or row['season'] == 4), 2)),
        axis=1
    )
    
    # Calculate solar irradiance
    df['solar_irradiance_base'] = df.apply(
        lambda row: max(0, 1000 * np.sin(np.pi * row['hour']/12) 
                       if 6 <= row['hour'] <= 18 else 0),
        axis=1
    )
    
    # Adjust irradiance for day of year (season)
    day_length_factor = df['day_of_year'].apply(
        lambda d: 0.7 + 0.6 * np.sin(2 * np.pi * (d - 172) / 365)
    )
    df['solar_irradiance'] = df['solar_irradiance_base'] * day_length_factor * (1 - 0.75 * df['cloud_cover'])
    
    # Calculate solar generation (kW)
    # Base generation under ideal conditions
    df['ideal_generation'] = df['solar_irradiance'] * panels_capacity_kw / 1000
    
    # Temperature efficiency factor (decrease efficiency as temperature increases above 25°C)
    temp_efficiency = df['temperature'].apply(lambda t: 1 - max(0, (t - 25) * 0.004))
    
    # Final generation with weather effects and random variation
    df['generation_kw'] = df['ideal_generation'] * temp_efficiency * (1 - df['cloud_cover']*0.8)
    
    # Add random noise and system inefficiencies
    df['generation_kw'] = df['generation_kw'] * np.random.normal(0.98, 0.05, len(df))
    df['generation_kw'] = df['generation_kw'].apply(lambda x: max(0, x))  # Ensure non-negative
    
    # Drop intermediate calculation columns
    df = df.drop(['temp_raw', 'solar_irradiance_base', 'ideal_generation', 'cloud_cover_base'], axis=1)
    
    return df
