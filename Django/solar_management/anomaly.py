# solar_management/model_utils.py
import joblib
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects # type: ignore
from django.conf import settings
from tensorflow.keras import layers # type: ignore

# Define the custom metric function
def mse(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1)

# Register the custom metric
get_custom_objects()['mse'] = mse

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
    
    def call(self, inputs, states):
        prev_output = states[0]
        h = tf.nn.tanh(tf.matmul(inputs, self.w) + 
                     tf.matmul(prev_output, self.u) + 
                     self.b)
        return h, [h]

# solar_management/utils/model_serving.py

import tensorflow as tf
import numpy as np
from django.conf import settings
# (Include your custom layers and other imports here)

class AnomalyDetector:
    def __init__(self):
        self.model = tf.keras.models.load_model(
            settings.BASE_DIR / 'anomaly_model.h5',
            custom_objects={'LTC': tf.keras.layers.RNN, 'LiquidLayer': LiquidLayer}
        )
        self.scaler = self._load_scaler('unified_scalar.save')  # if scaler is used

    def _load_scaler(self, path):
        """Create dummy scaler if file missing"""
        try:
            return joblib.load(settings.BASE_DIR / path)
        except FileNotFoundError:
            print("WARNING: Using emergency scaler with default values")
            scaler = StandardScaler()
            # Mock training with expected feature ranges
            scaler.mean_ = np.array([220.0, 5.0, 850.0, 0.0, 1.0])  # Voltage, Current, etc.
            scaler.scale_ = np.array([30.0, 2.0, 150.0, 1.0, 1.0])
            scaler.var_ = scaler.scale_ ** 2
            scaler.n_features_in_ = 5
            return scaler

    def preprocess(self, input_data):
        """
        For simplicity during testing, if you only have a single vector (5 features),
        you can replicate it 24 times to form a sequence.
        """
        # Convert input_data to a numpy array and reshape to (1, 5)
        data = np.array(input_data).reshape(1, 5)
        # Replicate the single time-step 24 times to simulate a sequence
        sequence = np.tile(data, (24, 1))  # now shape is (24, 5)
        # Add a batch dimension so the final shape is (1, 24, 5)
        return np.expand_dims(sequence, axis=0)

    def predict(self, data):
        processed = self.preprocess(data)
        predictions = self.model.predict(processed)
        return predictions.tolist()
