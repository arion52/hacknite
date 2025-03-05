# solar_management/admin.py
from django.contrib import admin
from .models import PowerData, PowerUsage, WeatherData, GenerationPrediction

admin.site.register(PowerUsage)

admin.site.register(PowerData)


@admin.register(WeatherData)
class WeatherDataAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'temperature', 'cloud_cover', 'wind_speed', 'solar_irradiance')
    search_fields = ('timestamp',)
    list_filter = ('timestamp',)

@admin.register(GenerationPrediction)
class GenerationPredictionAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'predicted_kw', 'actual_kw')
    search_fields = ('timestamp',)
    list_filter = ('timestamp',)
    raw_id_fields = ('weather_data',)
