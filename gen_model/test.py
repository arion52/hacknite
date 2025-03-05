import pickle
from tensorflow.keras.models import load_model

# Load the saved scalers
with open('generation_X_scaler.pkl', 'rb') as f:
    X_scaler = pickle.load(f)
with open('generation_y_scaler.pkl', 'rb') as f:
    y_scaler = pickle.load(f)

# Load the trained model
model = load_model('generation_model.h5')

# Prepare test data
features = ['temperature', 'cloud_cover', 'wind_speed', 'solar_irradiance', 
            'hour', 'day_of_year', 'month']
X_test_new = new_data[features].values
X_test_scaled = X_scaler.transform(X_test_new)

# Make predictions
y_pred_scaled = model.predict(X_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

# Add predictions to the dataframe for comparison
new_data['predicted_generation_kw'] = y_pred

# Save predictions for analysis
new_data.to_csv("solar_data_predictions.csv", sep='\t', encoding='utf-8', index=False)
