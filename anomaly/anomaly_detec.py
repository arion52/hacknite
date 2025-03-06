import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models
from sklearn.preprocessing import RobustScaler
from datetime import datetime, timedelta
from ncps import wirings
from ncps.keras import LTC
from tensorflow import keras

# 1. Enhanced Synthetic Data Generation with Realistic Patterns
def generate_solar_data(days=365, panels=10):
    np.random.seed(42)
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(minutes=15*i) for i in range(days*96)]
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'irradiance': np.abs(np.random.normal(0, 800, len(timestamps))),
        'voltage': np.zeros(len(timestamps)),
        'current': np.zeros(len(timestamps)),
        'is_raining': np.zeros(len(timestamps)),
        'is_daylight': np.zeros(len(timestamps))
    })
    
    # Generate realistic daily patterns
    for i, ts in enumerate(timestamps):
        hour = ts.hour + ts.minute/60
        daylight = 6 <= hour <= 18
        df.loc[i, 'is_daylight'] = int(daylight)
        
        # Base irradiance with seasonal variation
        day_of_year = ts.timetuple().tm_yday
        season_factor = 0.5 + 0.5*np.sin(2*np.pi*(day_of_year-80)/365)
        
        if daylight:
            df.loc[i, 'irradiance'] = 800 * season_factor * (1 - 0.5*np.random.rand())
            if np.random.rand() < 0.1:  # 10% chance of rain during daylight
                df.loc[i, 'is_raining'] = 1
                df.loc[i, 'irradiance'] *= 0.2 + 0.1*np.random.rand()
        else:
            df.loc[i, 'irradiance'] = 0
            
    # Physical model calculations
    df['voltage'] = 30 + 5*np.sin(2*np.pi*df.index/(96*7)) + np.random.normal(0, 0.5, len(df))
    df['current'] = df['irradiance'] * (0.4 + 0.1*np.random.rand()) / df['voltage']
    
    # Introduce synthetic anomalies (2% of data points)
    anomaly_indices = np.random.choice(len(df), int(0.02*len(df)), replace=False)
    for i in anomaly_indices:
        if df.loc[i, 'is_daylight'] and not df.loc[i, 'is_raining']:
            df.loc[i, 'voltage'] *= 0.2 + 0.3*np.random.rand()
            df.loc[i, 'current'] *= 0.1 + 0.2*np.random.rand()
    
    return df

def lnn_model():
    ncp_arch = wirings.AutoNCP(8, 1) # 8 input units and 1 output unit

    ncp_model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            LTC(ncp_arch, return_sequences=True),
        ]
    )

    ncp_model.compile(
        optimizer=keras.optimizers.Adam(0.01),  # Learning rate set to 0.01
        loss='mean_squared_error'  # Common loss function for regression tasks
    )
    ncp_model.summary()

# 2. Custom Liquid Neural Network Layer
class LiquidLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.output_size = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                initializer='glorot_uniform',
                                trainable=True)
        self.u = self.add_weight(shape=(self.units, self.units),
                                initializer='orthogonal',
                                trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                initializer='zeros',
                                trainable=True)
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = tf.nn.tanh(tf.matmul(inputs, self.w) + 
                      tf.matmul(prev_output, self.u) + 
                      self.b)
        return h, [h]

# 3. Hybrid LNN-LSTM Model Architecture
def build_hybrid_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Liquid Neural Network Section
    x = layers.RNN(LiquidLayer(64), return_sequences=True)(inputs)
    x = layers.LayerNormalization()(x)
    
    # Temporal Feature Extraction
    x = layers.LSTM(128, return_sequences=True, recurrent_dropout=0.2)(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    
    # Contextual Attention
    x = layers.Dense(32, activation='selu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output Layer
    outputs = layers.Dense(input_shape[-1], activation='linear')(x)
    
    model = models.Model(inputs, outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss='mse',
                metrics=['mae'])
    
    return model

# 4. Data Preprocessing Pipeline
def preprocess_data(df, seq_length=24):
    # Feature Selection
    features = ['voltage', 'current', 'irradiance', 'is_raining', 'is_daylight']
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    # Sequence Creation
    X, y = [], []
    for i in range(len(scaled_data)-seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Train-Test Split
    split = int(0.8*len(X))
    return X[:split], y[:split], X[split:], y[split:], scaler

# 5. Visualization and Analysis
def plot_time_series(df):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(df.index, df['voltage'], label="Actual Voltage", color="blue")
    axes[0].plot(df.index, df['pred_voltage'], label="Predicted Voltage", linestyle="dashed", color="red")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].legend()
    
    axes[1].plot(df.index, df['current'], label="Actual Current", color="green")
    axes[1].plot(df.index, df['pred_current'], label="Predicted Current", linestyle="dashed", color="red")
    axes[1].set_ylabel("Current (A)")
    axes[1].legend()

    axes[2].plot(df.index, df['irradiance'], label="Actual Irradiance", color="orange")
    axes[2].plot(df.index, df['pred_irradiance'], label="Predicted Irradiance", linestyle="dashed", color="red")
    axes[2].set_ylabel("Irradiance (W/m²)")
    axes[2].set_xlabel("Time")
    axes[2].legend()

    plt.suptitle("Actual vs Predicted Solar Panel Data")
    plt.show()

def plot_anomaly_scatter(df):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=df['voltage'], y=df['current'], hue=df['is_raining'], palette="coolwarm", alpha=0.7)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title("Voltage vs Current - Highlighting Rain Conditions")
    plt.legend(title="Rain (1=Yes, 0=No)")
    plt.show()

# 6. End-to-End Training and Validation
def main():
    # Generate and prepare data
    solar_df = generate_solar_data(days=180)
    X_train, y_train, X_test, y_test, scaler = preprocess_data(solar_df)
    
    # Build and train model
    model = build_hybrid_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train,
              epochs=50,
              batch_size=64,
              validation_split=0.2,
              verbose=1)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Save predictions
    results_df = pd.DataFrame(y_test, columns=['voltage', 'current', 'irradiance', 'is_raining', 'is_daylight'])
    results_df[['pred_voltage', 'pred_current', 'pred_irradiance', 'pred_is_raining', 'pred_is_daylight']] = y_pred
    results_df.to_csv('solar_predictions.csv', index=False)

    # Plot Results
    plot_time_series(results_df)
    plot_anomaly_scatter(results_df)

if __name__ == "__main__":
    main()
