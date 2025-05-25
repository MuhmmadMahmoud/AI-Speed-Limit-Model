from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import os
import requests
import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# API Configuration with defaults
OPENSTREETMAP_USER_AGENT = os.getenv('OPENSTREETMAP_USER_AGENT', 'SpeedLimitApp/1.0')
OPENMETEO_API_URL = os.getenv('OPENMETEO_API_URL', 'https://api.open-meteo.com/v1/forecast')
OVERPASS_API_URL = os.getenv('OVERPASS_API_URL', 'https://overpass-api.de/api/interpreter')

# Load the trained model
try:
    model_path = 'speed_limit_model_saved'
    model = tf.saved_model.load(model_path)
    predict_fn = model.signatures['serving_default']
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Load the scaler
try:
    import pickle
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded successfully")
except FileNotFoundError:
    logger.warning("Scaler file not found, proceeding without scaling")
    scaler = None
except Exception as e:
    logger.error(f"Error loading scaler: {e}")
    scaler = None

# Function to get road type from coordinates
def get_road_type(latitude, longitude):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}"
        response = requests.get(url, headers={'User-Agent': OPENSTREETMAP_USER_AGENT}, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if 'road' in data.get('address', {}):
            road_name = data['address']['road']
            if 'highway' in road_name.lower() or 'freeway' in road_name.lower() or 'motorway' in road_name.lower():
                return 1  # Highway
            elif 'street' in road_name.lower() or 'avenue' in road_name.lower():
                return 2  # Urban
        return 0  # Default to residential
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting road type: {e}")
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
        url = f"{OPENMETEO_API_URL}?latitude={latitude}&longitude={longitude}&current=precipitation,rain,showers,snowfall,weathercode"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        current = data.get('current', {})
        weather_code = current.get('weathercode', 0)
        precipitation = current.get('precipitation', 0)
        
        if weather_code >= 95:  # Thunderstorm
            return 2
        elif weather_code >= 51 or precipitation > 0.5:  # Rain or drizzle
            return 2
        elif weather_code in [45, 48]:  # Fog
            return 1
        return 0  # Clear
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting weather: {e}")
        return 0  # Default to clear weather on error

# Function to check proximity to schools
def check_school_proximity(latitude, longitude):
    try:
        radius = 500  # meters
        overpass_query = f"""
        [out:json];
        node["amenity"="school"](around:{radius},{latitude},{longitude});
        out;
        """
        response = requests.post(OVERPASS_API_URL, data=overpass_query, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        return 1 if len(data.get('elements', [])) > 0 else 0
    except requests.exceptions.RequestException as e:
        logger.error(f"Error checking school proximity: {e}")
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
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract coordinates and speed
        try:
            latitude = float(data.get('latitude', 0))
            longitude = float(data.get('longitude', 0))
            actual_speed = float(data.get('actual_speed', 0))
        except (TypeError, ValueError) as e:
            return jsonify({'error': 'Invalid input values'}), 400
        
        # Validate coordinates
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            return jsonify({'error': 'Invalid coordinates'}), 400
        
        # Get road and environmental features
        road_type = get_road_type(latitude, longitude)
        traffic = estimate_traffic(latitude, longitude)
        curvature = estimate_curvature(latitude, longitude)
        weather = get_weather(latitude, longitude)
        proximity_to_school = check_school_proximity(latitude, longitude)
        time_of_day = get_time_of_day()
        
        # Prepare input for model
        input_data = np.array([[road_type, traffic, curvature, weather, 
                               proximity_to_school, time_of_day]], dtype=np.float32)
        
        # Apply scaling if available
        if scaler:
            input_data = scaler.transform(input_data)
        
        # Make prediction
        try:
            input_tensor = tf.convert_to_tensor(input_data)
            predictions = predict_fn(input_tensor)
            predicted_speed_limit = float(predictions['dense_3'].numpy()[0][0])
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return jsonify({'error': 'Error making prediction'}), 500
        
        # Round to nearest 10 for more realistic speed limit
        allowed_speed = round(predicted_speed_limit / 10) * 10
        
        # Prepare response
        response = {
            'allowed_speed': allowed_speed,
            'features': {
                'road_type': road_type,
                'traffic': traffic,
                'curvature': curvature,
                'weather': weather,
                'proximity_to_school': proximity_to_school,
                'time_of_day': time_of_day
            }
        }
        
        # Add warning if exceeding speed limit
        if actual_speed > allowed_speed:
            speed_diff = actual_speed - allowed_speed
            response['warning'] = f"Over speed by {speed_diff} km/h"
        else:
            response['warning'] = "Speed within limit"
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)