# check_files.py
import os

# Define the paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'generation_model.h5')
x_scaler_path = os.path.join(base_dir, 'generation_X_scaler.pkl')
y_scaler_path = os.path.join(base_dir, 'generation_y_scaler.pkl')

# Check if files exist
print(f"Checking for model file at: {model_path}")
print(f"Exists: {os.path.exists(model_path)}")

print(f"\nChecking for X scaler file at: {x_scaler_path}")
print(f"Exists: {os.path.exists(x_scaler_path)}")

print(f"\nChecking for Y scaler file at: {y_scaler_path}")
print(f"Exists: {os.path.exists(y_scaler_path)}")
