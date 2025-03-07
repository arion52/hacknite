from datetime import datetime, timedelta
from django.http import JsonResponse, HttpResponse
from .models import PowerData
from .utils import UnifiedUsageHandler, get_weather_forecast, check_if_panel_defective, calculate_mppt

from django.conf import settings
import numpy as np
import json
import pickle
from django.views.decorators.http import require_http_methods
from rest_framework.permissions import AllowAny
from rest_framework.decorators import api_view, permission_classes
from .utils import GenerationModelHandler, generate_sample_solar_data
from .models import WeatherData, GenerationPrediction
from django.utils import timezone
from .models import UsagePrediction, PowerUsage
import pandas as pd
from rest_framework.response import Response
from .anomaly import AnomalyDetector
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import AnomalyRequestSerializer

# Returns weather forecast suggestion
@api_view(['GET'])
@permission_classes([AllowAny])  # Allow unauthenticated access
def weather_advice(request):
    suggestion = get_weather_forecast()
    return JsonResponse({'suggestion': suggestion})

# Checks the status of a solar panel
@api_view(['GET'])
def check_panel_status(request, panel_id):
    panel_data = {
        'input_power': -1  # Simulated data
    }
    defective = check_if_panel_defective(panel_data)
    return JsonResponse({'panel_id': panel_id, 'defective': defective})

# Retrieves power data for a sensor
@api_view(['GET'])
def get_power_data(request, sensor_id):
    power_data = PowerData.objects.filter(sensor_id=sensor_id)
    response_data = [{
        'date': str(data.date),
        'input_power': data.input_power,
        'usage_power': data.usage_power
    } for data in power_data]
    return JsonResponse(response_data, safe=False)

# Default view for the app
def index(request):
    return HttpResponse("Welcome to the Solar Management App!")

# ANOMALY DETECTION MODEL

detector = AnomalyDetector()  # Loaded once at server start

@api_view(['POST'])
def anomaly_detection(request):
    serializer = AnomalyRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)
    
    raw_data = serializer.validated_data
    input_array = [
        raw_data['voltage'],
        raw_data['current'],
        raw_data['irradiance'],
        int(raw_data['is_raining']),
        int(raw_data['is_daylight'])
    ]
    
    prediction = detector.predict([input_array])
    return Response({
        'anomaly_score': prediction[0][0],
        'threshold': 0.85,
        'is_anomalous': prediction[0][0] > 0.85
    })

# GENERATION PREDICTION MODEL
generation_model_handler = GenerationModelHandler.get_instance()

# Replace the existing predict_generated function with this implementation
@require_http_methods(["POST"])
def predict_generated(request):
    """API endpoint for solar generation prediction using TensorFlow model"""
    try:
        # Parse the request data
        data = json.loads(request.body)
        
        # Get the input data
        if 'features' in data:
            # Format 1: Dictionary of features
            features = data['features']
            input_data = np.array([[
                features.get('temperature', 0),
                features.get('cloud_cover', 0),
                features.get('wind_speed', 0),
                features.get('solar_irradiance', 0),
                features.get('hour', datetime.now().hour),
                features.get('day_of_year', datetime.now().timetuple().tm_yday),
                features.get('month', datetime.now().month)
            ]])
        elif 'input_data' in data and isinstance(data['input_data'], list):
            # Format 2: Array of values [temp, cloud, wind, irradiance, hour, day, month]
            input_data = np.asarray(data['input_data'])
            if len(input_data.shape) == 1:
                input_data = input_data.reshape(1, -1)
        else:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid input format. Provide either "features" dict or "input_data" array'
            }, status=400)
        
        # Check input shape
        if input_data.shape[1] != 7:
            return JsonResponse({
                'status': 'error',
                'message': f'Expected 7 features, but got {input_data.shape[1]}'
            }, status=400)
        
        # Make prediction
        prediction = generation_model_handler.predict(input_data)
        predicted_kw = float(prediction[0][0])
        
        # Save to database if timestamp is provided
        if 'timestamp' in data:
            timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            
            # Create weather data entry
            weather_data = WeatherData.objects.create(
                timestamp=timestamp,
                temperature=input_data[0][0],
                cloud_cover=input_data[0][1],
                wind_speed=input_data[0][2],
                solar_irradiance=input_data[0][3]
            )
            
            # Create prediction entry
            GenerationPrediction.objects.create(
                weather_data=weather_data,
                predicted_kw=predicted_kw,
                timestamp=timestamp
            )
        
        # Return the prediction
        return JsonResponse({
            'status': 'success',
            'input': input_data.tolist(),
            'prediction': predicted_kw,
            'units': 'kW'
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid JSON format'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

# Add a new view for generating a forecast
@require_http_methods(["GET"])
def generate_forecast(request):
    """Generate a solar generation forecast for the next 24 hours"""
    try:
        # Generate sample weather data for the next 24 hours
        sample_data = generate_sample_solar_data(days=1)
        
        # Make predictions
        predictions = generation_model_handler.predict(sample_data)
        
        # Prepare the forecast data
        forecast_data = []
        for i, (_, row) in enumerate(sample_data.iterrows()):
            forecast_data.append({
                'timestamp': row['datetime'].isoformat(),
                'hour': int(row['hour']),
                'temperature': float(row['temperature']),
                'cloud_cover': float(row['cloud_cover']),
                'wind_speed': float(row['wind_speed']),
                'solar_irradiance': float(row['solar_irradiance']),
                'predicted_kw': float(predictions[i][0])
            })
        
        return JsonResponse({
            'status': 'success',
            'forecast': forecast_data
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

# USAGE DETECTION MODEL
usage_handler = UnifiedUsageHandler.get_instance()

@require_http_methods(["POST"])
def predict_usage(request):
    """API endpoint for unified usage prediction"""
    try:
        data = json.loads(request.body)
        
        # Validate required fields
        required_fields = ['timestamp', 'consumer_type', 'global_active_power']
        if not all(field in data for field in required_fields):
            return JsonResponse({
                'status': 'error',
                'message': f'Missing required fields. Required: {required_fields}'
            }, status=400)
        
        # Add default values from feature config
        data.setdefault('cloud_cover', 0)
        data.setdefault('temperature', 25.0)
        
        # Make prediction
        prediction = usage_handler.predict([data])
        
        # Save to database
        UsagePrediction.objects.create(
            timestamp=datetime.fromisoformat(data['timestamp']),
            predicted_kw=prediction,
            consumer_type=data['consumer_type'],
            actual_kw=None
        )
        
        return JsonResponse({
            'status': 'success',
            'prediction': prediction,
            'units': 'kW',
            'model': 'unified'
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@api_view(['GET'])
def mppt_view(request):
    voltage = float(request.GET.get('voltage', 0))
    current = float(request.GET.get('current', 0))
    prev_voltage = float(request.GET.get('prev_voltage', 0))
    prev_current = float(request.GET.get('prev_current', 0))

    next_voltage, max_power = calculate_mppt(voltage, current, prev_voltage, prev_current)
    return JsonResponse({
        'next_voltage': next_voltage,
        'max_power': max_power
    })

@require_http_methods(["POST"])
def post_sensor_data(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            voltage = data.get('voltage')
            current = data.get('current')

            # Process or store the data as needed
            print(f"Received voltage: {voltage}, current: {current}")

            return JsonResponse({'status': 'success', 'message': 'Data received successfully'})
        
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'status': 'error', 'message': 'Only POST requests are allowed'}, status=405)

from .models import PowerUsage

def predict_usage_from_db(request):
    # Fetch recent power usage data from the database
    latest_data = PowerUsage.objects.order_by('-dt').first()

    if latest_data:
        input_data = [
            latest_data.global_active_power,
            latest_data.global_reactive_power,
            latest_data.voltage,
            latest_data.global_intensity,
            latest_data.sub_metering_1,
            latest_data.sub_metering_2,
            latest_data.sub_metering_3,
        ]

        # Call the usage prediction model with the input data
        input_data = np.asarray(input_data).reshape(1, 1, 7)
        prediction = usage_handler.predict(input_data)

        return JsonResponse({
            'input': input_data.tolist(),
            'prediction': prediction.tolist(),
            'status': 'success'
        })

    return JsonResponse({'status': 'error', 'message': 'No data available'}, status=400)
