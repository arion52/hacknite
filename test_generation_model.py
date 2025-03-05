# test_generation_model.py
import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'solar_management.settings')
django.setup()

# Imports after Django setup
import numpy as np
from solar_management.utils import GenerationModelHandler

def test_model():
    print("Testing the generation model...")
    
    try:
        # Get model instance
        model = GenerationModelHandler.get_instance()
        
        # Test with simple data: [temp, cloud, wind, irradiance, hour, day, month]
        test_data = np.array([[25.0, 0.2, 4.5, 800, 12, 172, 6]])
        
        # Make a prediction
        prediction = model.predict(test_data)
        print(f"\nPrediction for sunny day: {prediction[0][0]:.2f} kW")
        
        # Test with more examples
        test_data_batch = np.array([
            [25.0, 0.2, 4.5, 800, 12, 172, 6],   # Sunny summer day
            [15.0, 0.8, 3.0, 300, 8, 80, 3],     # Cloudy spring morning
            [5.0, 0.1, 6.0, 400, 16, 355, 12]    # Clear winter afternoon
        ])
        
        # Make batch predictions
        predictions = model.predict(test_data_batch)
        
        print("\nBatch predictions:")
        for i, pred in enumerate(predictions):
            print(f"  Example {i+1}: {pred[0]:.2f} kW")
            
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
