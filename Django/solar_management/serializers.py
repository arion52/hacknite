# solar_management/api/serializers.py
from rest_framework import serializers

class AnomalyRequestSerializer(serializers.Serializer):
    voltage = serializers.FloatField()
    current = serializers.FloatField()
    irradiance = serializers.FloatField()
    is_raining = serializers.BooleanField()
    is_daylight = serializers.BooleanField()
    timestamp = serializers.DateTimeField()
