# solar_management/model_utils.py
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects # type: ignore

# Define the custom metric function
def mse(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1)

# Register the custom metric
get_custom_objects()['mse'] = mse

class ModelHandler:
    def __init__(self, model_path):
        # Load the model and pass custom_objects to handle custom metrics
        self.model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'mse': mse}
        )

    def predict(self, input_data):
        return self.model.predict(input_data)