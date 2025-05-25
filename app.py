from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import os
import requests
import datetime
import time
import gc
from functools import lru_cache

app = Flask(__name__)

# Load the trained model
model_path = 'speed_limit_model.h5'
model = load_model(model_path)

# Load the scaler if you have one
import pickle
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    # If scaler file doesn't exist, proceed without it
    scaler = None

# Cache for API responses
road_cache = {}
weather_cache = {}

@lru_cache(maxsize=100)
def get_road_data(latitude, longitude):
    cache_key = f"{latitude:.4f}_{longitude:.4f}"
    if cache_key in road_cache:
        if time.time() - road_cache[cache_key]['timestamp'] < 1800:  # 30 min cache
            return road_cache[cache_key]['data']
    
    try:
        # Using Overpass API for better road data
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json];
        way(around:50,{latitude},{longitude})[highway];
        out geom;
        """
        response = requests.get(overpass_url, params={'data': query})
        data = response.json()
        
        if not data['elements']:
            road_type = 0  # Default to residential
        else:
            road = data['elements'][0]
            tags = road.get('tags', {})
            road_type = tags.get('highway', 'unknown')
            
            # Map to our model's road types (0: residential, 1: highway, 2: urban)
            if road_type in ['motorway', 'trunk', 'primary']:
                road_type = 1  # Highway
            elif road_type in ['secondary', 'tertiary']:
                road_type = 2  # Urban
            else:
                road_type = 0  # Residential
                
        result = {'road_type': road_type}
        road_cache[cache_key] = {
            'data': result,
            'timestamp': time.time()
        }
        return result
    except Exception as e:
        print(f"Error getting road data: {e}")
        return {'road_type': 0}  # Default to residential on error

@lru_cache(maxsize=100)
def get_weather_data(latitude, longitude):
    cache_key = f"{latitude:.4f}_{longitude:.4f}"
    if cache_key in weather_cache:
        if time.time() - weather_cache[cache_key]['timestamp'] < 1800:  # 30 min cache
            return weather_cache[cache_key]['data']
    
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,weathercode,precipitation"
        response = requests.get(url)
        data = response.json()
        
        current = data.get('current', {})
        weather_code = current.get('weathercode', 0)
        temperature = current.get('temperature_2m', 20)
        precipitation = current.get('precipitation', 0)
        
        # Classify weather (0: clear, 1: foggy, 2: rainy)
        if weather_code >= 95 or precipitation > 0.5:  # Thunderstorm or heavy rain
            weather = 2
        elif weather_code >= 51 or precipitation > 0.1:  # Light rain
            weather = 2
        elif weather_code in [45, 48]:  # Fog
            weather = 1
        else:
            weather = 0  # Clear
            
        result = {'weather': weather}
        weather_cache[cache_key] = {
            'data': result,
            'timestamp': time.time()
        }
        return result
    except Exception as e:
        print(f"Error getting weather: {e}")
        return {'weather': 0}  # Default to clear weather

def estimate_traffic(latitude, longitude):
    # This is a simplified traffic estimation
    # In a real app, you would use a traffic API
    current_hour = datetime.datetime.now().hour
    
    # Rush hours typically have higher traffic
    if (current_hour >= 7 and current_hour <= 9) or (current_hour >= 16 and current_hour <= 18):
        return 0.8  # High traffic during rush hours
    elif (current_hour >= 10 and current_hour <= 15) or (current_hour >= 19 and current_hour <= 21):
        return 0.5  # Medium traffic during day/evening
    else:
        return 0.2  # Low traffic at night/early morning

def check_school_proximity(latitude, longitude):
    try:
        # Using OpenStreetMap Overpass API to find nearby schools
        # This is a simplified version
        radius = 500  # meters
        overpass_url = "https://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
        node["amenity"="school"](around:{radius},{latitude},{longitude});
        out;
        """
        response = requests.post(overpass_url, data=overpass_query)
        data = response.json()
        
        # If any schools are found within radius
        if len(data.get('elements', [])) > 0:
            return 1
        else:
            return 0
    except Exception as e:
        print(f"Error checking school proximity: {e}")
        return 0  # Default to no schools nearby on error

def get_time_of_day():
    current_hour = datetime.datetime.now().hour
    
    if current_hour >= 5 and current_hour < 12:
        return 0  # Morning
    elif current_hour >= 12 and current_hour < 17:
        return 1  # Afternoon
    elif current_hour >= 17 and current_hour < 21:
        return 2  # Evening
    else:
        return 3  # Night

def estimate_curvature(latitude, longitude):
    try:
        # Using Overpass API to get road geometry
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json];
        way(around:50,{latitude},{longitude})[highway];
        out geom;
        """
        response = requests.get(overpass_url, params={'data': query})
        data = response.json()
        
        if not data['elements']:
            return 0.3  # Default moderate curvature
            
        road = data['elements'][0]
        geometry = road.get('geometry', [])
        
        if len(geometry) < 3:
            return 0.1  # Straight road
            
        # Calculate total angle change
        total_angle = 0.0
        for i in range(1, len(geometry) - 1):
            p1 = geometry[i - 1]
            p2 = geometry[i]
            p3 = geometry[i + 1]
            
            # Calculate vectors
            v1 = (p1['lon'] - p2['lon'], p1['lat'] - p2['lat'])
            v2 = (p3['lon'] - p2['lon'], p3['lat'] - p2['lat'])
            
            # Calculate angle
            dot_prod = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = (v1[0]**2 + v1[1]**2)**0.5
            mag2 = (v2[0]**2 + v2[1]**2)**0.5
            
            if mag1 * mag2 == 0:
                continue
                
            cos_angle = dot_prod / (mag1 * mag2)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
            angle = abs(np.degrees(np.arccos(cos_angle)))
            total_angle += angle
            
        # Normalize curvature to 0-1 range
        if total_angle < 10:
            return 0.1  # Straight
        elif total_angle < 30:
            return 0.3  # Moderate curve
        else:
            return 0.5  # Sharp curve
            
    except Exception as e:
        print(f"Error calculating curvature: {e}")
        return 0.3  # Default moderate curvature

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Speed limit prediction API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        latitude = data.get('latitude', 0)
        longitude = data.get('longitude', 0)
        actual_speed = data.get('actual_speed', 0)
        
        # Get enhanced road and weather data
        road_info = get_road_data(latitude, longitude)
        weather_info = get_weather_data(latitude, longitude)
        
        # Get other features
        traffic = estimate_traffic(latitude, longitude)
        curvature = estimate_curvature(latitude, longitude)
        proximity_to_school = check_school_proximity(latitude, longitude)
        time_of_day = get_time_of_day()
        
        # Prepare input for model using all features
        input_data = np.array([[
            road_info['road_type'],
            traffic,
            curvature,
            weather_info['weather'],
            proximity_to_school,
            time_of_day
        ]])
        
        # Apply scaler if available
        if scaler:
            input_data = scaler.transform(input_data)
        
        # Make prediction
        predicted_speed_limit = float(model.predict(input_data, verbose=0)[0][0])
        
        # Round to nearest 10 for more realistic speed limit
        allowed_speed = round(predicted_speed_limit / 10) * 10
        
        # Keep the same response format
        response = {
            'allowed_speed': allowed_speed
        }
        
        # Add warning if exceeding speed limit
        if actual_speed > allowed_speed:
            speed_diff = actual_speed - allowed_speed
            response['warning'] = f"Over speed by {speed_diff} km/h"
        else:
            response['warning'] = "Speed within limit"
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Add memory cleanup
@app.after_request
def after_request(response):
    gc.collect()
    tf.keras.backend.clear_session()
    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)