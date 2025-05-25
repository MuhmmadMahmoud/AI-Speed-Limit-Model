from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import os
import requests
import datetime

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

# Function to get road type from coordinates
def get_road_type(latitude, longitude):
    try:
        # Using OpenStreetMap Nominatim API to get road information
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}"
        response = requests.get(url, headers={'User-Agent': 'SpeedLimitApp/1.0'})
        data = response.json()
        
        # Extract road type information
        if 'road' in data.get('address', {}):
            road_name = data['address']['road']
            # Simple classification based on road name
            if 'highway' in road_name.lower() or 'freeway' in road_name.lower() or 'motorway' in road_name.lower():
                return 1  # Highway
            elif 'street' in road_name.lower() or 'avenue' in road_name.lower():
                return 2  # Urban
            else:
                return 0  # Residential
        return 0  # Default to residential if no road info
    except Exception as e:
        print(f"Error getting road type: {e}")
        return 0  # Default to residential on error

# Function to estimate traffic based on time and location
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

# Function to get weather conditions
def get_weather(latitude, longitude):
    try:
        # Using Open-Meteo API for weather data
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=precipitation,rain,showers,snowfall,weathercode"
        response = requests.get(url)
        data = response.json()
        
        # Check weather conditions
        current = data.get('current', {})
        weather_code = current.get('weathercode', 0)
        precipitation = current.get('precipitation', 0)
        
        # Classify weather (0: clear, 1: foggy, 2: rainy)
        if weather_code >= 95:  # Thunderstorm
            return 2
        elif weather_code >= 51 or precipitation > 0.5:  # Rain or drizzle
            return 2
        elif weather_code in [45, 48]:  # Fog
            return 1
        else:
            return 0  # Clear
    except Exception as e:
        print(f"Error getting weather: {e}")
        return 0  # Default to clear weather on error

# Function to check proximity to schools
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

# Function to determine time of day
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

# Function to estimate road curvature
def estimate_curvature(latitude, longitude):
    # This is a placeholder for road curvature estimation
    # In a real app, you would use map data to calculate actual curvature
    # For now, we'll return a random value between 0.1 and 0.5
    return 0.3

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Speed limit prediction API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract coordinates and speed
        latitude = data.get('latitude', 0)
        longitude = data.get('longitude', 0)
        actual_speed = data.get('actual_speed', 0)
        
        # Get road and environmental features from coordinates
        road_type = get_road_type(latitude, longitude)
        traffic = estimate_traffic(latitude, longitude)
        curvature = estimate_curvature(latitude, longitude)
        weather = get_weather(latitude, longitude)
        proximity_to_school = check_school_proximity(latitude, longitude)
        time_of_day = get_time_of_day()
        
        # Prepare input for model
        input_data = np.array([[road_type, traffic, curvature, weather, 
                               proximity_to_school, time_of_day]])
        
        # If you have a scaler, apply it here
        if scaler:
            input_data = scaler.transform(input_data)
        
        # Make prediction
        predicted_speed_limit = float(model.predict(input_data)[0][0])
        
        # Round to nearest 10 for more realistic speed limit
        allowed_speed = round(predicted_speed_limit / 10) * 10
        
        # Prepare response
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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)