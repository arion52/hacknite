import os
import pickle
from django.conf import settings
from django.db import models
import numpy as np
from tensorflow.keras.models import load_model


class PowerData(models.Model):
    sensor_id = models.CharField(max_length=100)
    date = models.DateField()
    input_power = models.FloatField()
    usage_power = models.FloatField()

class PowerUsage(models.Model):
    dt = models.DateTimeField()
    global_active_power = models.FloatField()
    global_reactive_power = models.FloatField()
    voltage = models.FloatField()
    global_intensity = models.FloatField()
    sub_metering_1 = models.FloatField()
    sub_metering_2 = models.FloatField()
    sub_metering_3 = models.FloatField()

    def __str__(self):
        return f"{self.dt} - Active Power: {self.global_active_power}"
    
# solar_management/models.py

from django.db import models
from django.utils import timezone

class WeatherData(models.Model):
    """Model to store weather data for generation predictions"""
    timestamp = models.DateTimeField(default=timezone.now)
    temperature = models.FloatField(help_text="Temperature in Celsius")
    cloud_cover = models.FloatField(help_text="Cloud cover (0-1 scale)")
    wind_speed = models.FloatField(help_text="Wind speed in m/s")
    solar_irradiance = models.FloatField(help_text="Solar irradiance in W/m²")
    
    def __str__(self):
        return f"Weather data at {self.timestamp}"
    
    class Meta:
        verbose_name = "Weather Data"
        verbose_name_plural = "Weather Data"
        ordering = ['-timestamp']

class GenerationPrediction(models.Model):
    """Model to store solar generation predictions"""
    weather_data = models.ForeignKey(
        WeatherData, 
        on_delete=models.CASCADE,
        related_name='predictions'
    )
    predicted_kw = models.FloatField(help_text="Predicted solar generation in kW")
    actual_kw = models.FloatField(null=True, blank=True, help_text="Actual solar generation in kW (if available)")
    timestamp = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"Generation prediction: {self.predicted_kw:.2f}kW at {self.timestamp}"
    
    class Meta:
        verbose_name = "Generation Prediction"
        verbose_name_plural = "Generation Predictions"
        ordering = ['-timestamp']

class UsagePrediction(models.Model):
    """Model to store energy usage predictions"""
    timestamp = models.DateTimeField(default=timezone.now)
    predicted_kw = models.FloatField(help_text="Predicted energy usage in kW")
    actual_kw = models.FloatField(null=True, blank=True, help_text="Actual energy usage in kW (if available)")
    consumer_type = models.CharField(max_length=50, default="Residential", 
                                    help_text="Type of consumer (Residential/Business)")
    
    # Optional relation to weather data
    weather_data = models.ForeignKey(
        WeatherData, 
        on_delete=models.SET_NULL,
        related_name='usage_predictions',
        null=True,
        blank=True
    )
    
    def __str__(self):
        return f"Usage prediction: {self.predicted_kw:.2f}kW at {self.timestamp}"
    
    class Meta:
        verbose_name = "Usage Prediction"
        verbose_name_plural = "Usage Predictions"
        ordering = ['-timestamp']

