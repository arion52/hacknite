import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
import os

def mse(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1)

get_custom_objects()['mse'] = mse

class UsageModelLoader:
    def __init__(self, model_paths):
        self.model_paths = model_paths
        self.models = {}

    def load_model(self, consumer_type):
        if consumer_type not in self.models:
            if consumer_type not in self.model_paths:
                raise ValueError(f"Unknown consumer_type: {consumer_type}")
            self.models[consumer_type] = tf.keras.models.load_model(
                self.model_paths[consumer_type],
                custom_objects={'mse': mse}
            )
        return self.models[consumer_type]

    def predict(self, input_data, consumer_type):
        model = self.load_model(consumer_type)
        if len(input_data.shape) != 3:
            raise ValueError(f"Expected input shape (batch_size, time_steps, features), got {input_data.shape}")
        return model.predict(input_data)
