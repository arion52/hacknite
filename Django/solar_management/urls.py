from django.contrib import admin
from django.urls import path
from django.views.generic.base import RedirectView
from .views import predict_usage_from_db, weather_advice, check_panel_status, get_power_data, predict_view, predict_generated, predict_usage, mppt_view, post_sensor_data, generate_forecast
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('weather_advice/', weather_advice, name='weather_advice'),
    path('check_panel/<str:panel_id>/', check_panel_status, name='check_panel_status'),
    path('power_data/<str:sensor_id>/', get_power_data, name='get_power_data'),
    path('predict_anomaly/', predict_view, name='predict_view'),
    path('predict_generation/', predict_generated, name='predict_generated'),
    path('predict_usage/', predict_usage, name='predict_usage'),
    path('calculate_mppt/', mppt_view, name='mppt_view'),
    path('post_sensor_data/', post_sensor_data, name='post_sensor_data'), 
    path('', RedirectView.as_view(url='weather_advice/')),
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('predict_usage_from_db/', predict_usage_from_db, name='predict_usage_from_db'),
    path('generate_forecast/', generate_forecast, name='generate_forecast'),
]
