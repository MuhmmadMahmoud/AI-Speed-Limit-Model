#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load and preprocess data
file_path = "/content/speed_data.csv"
df = pd.read_csv(file_path)

label_encoders = {}
categorical_columns = ["Road_Type", "Traffic_Density", "Curvature", "Weather", "Time_of_Day"]
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df["Proximity_to_School"] = df["Proximity_to_School"].astype(int)

# Features (excluding Actual_Speed, Latitude, and Longitude)
features = ["Road_Type", "Traffic_Density", "Curvature", "Weather", "Proximity_to_School", "Time_of_Day"]
target = "Speed_Limit"

X = df[features].values
y = df[target].values

# Scale the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple feedforward neural network (DNN)
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))  # Dropout layer to avoid overfitting
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1))  # Output layer for regression (predicting Speed_Limit)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model with early stopping to avoid overfitting
epochs = 100
patience = 20
best_loss = float('inf')
counter = 0

for epoch in range(epochs):
    history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
    loss = history.history['loss'][0]

    if loss < best_loss:
        best_loss = loss
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered!")
        break

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {test_loss:.4f}")

# Predict and calculate MAE, RMSE
y_test_predicted = model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, y_test_predicted)
rmse = np.sqrt(mean_squared_error(y_test, y_test_predicted))

accuracy = 100 - (mae / np.mean(y_test) * 100)

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, Accuracy Estimate: {accuracy:.2f}%")

# Optional: Add a plot of predictions vs actual
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Expected', color='red')
plt.plot(y_test_predicted, label='Predicted', color='blue')
plt.title(f'Traffic Prediction\nOne day prediction (MSE = {mean_squared_error(y_test, y_test_predicted):.4f})')
plt.xlabel('Sample')
plt.ylabel('Velocity')
plt.legend()
plt.grid(True)
plt.show()



def predict_speed_limit_dnn(road_type, traffic, curvature, weather, proximity_to_school, time_of_day, actual_speed):

    input_data = np.array([[road_type, traffic, curvature, weather, proximity_to_school, time_of_day]])
    input_data = scaler.transform(input_data)


    predicted_speed_limit = model.predict(input_data)[0][0]

    print(f"\n السرعة الفعلية: {actual_speed} كم/س")
    print(f" السرعة المتوقعة (DNN): {predicted_speed_limit:.2f} كم/س")

    if actual_speed > predicted_speed_limit:
        print(" تحذير: تجاوز السرعة!")
    else:
        print(" السرعة مناسبة.")


predict_speed_limit_dnn(
    road_type=1,
    traffic=2,
    curvature=0,
    weather=1,
    proximity_to_school=0,
    time_of_day=3,
    actual_speed=30
    )


# In[ ]:


import requests
import math
from datetime import datetime

def get_road_data(lat, lon):
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    way(around:50,{lat},{lon})[highway];
    out geom;
    """
    response = requests.get(overpass_url, params={'data': query})
    data = response.json()
    if not data['elements']:
        return {
            'road_type': 'unknown',
            'school_nearby': False,
            'traffic_estimate': 'unknown',
            'curvature': 'unknown'
        }
    road = data['elements'][0]
    tags = road.get('tags', {})
    geometry = road.get('geometry', [])
    road_type = tags.get('highway', 'unknown')
    traffic_map = {
        'motorway': 'Low',
        'trunk': 'Medium',
        'primary': 'High',
        'secondary': 'Medium',
        'tertiary': 'Medium',
        'residential': 'Medium',
        'service': 'Low',
        'track': 'Low',
    }
    traffic_estimate = traffic_map.get(road_type, 'Medium')
    def calculate_curvature(geometry):
        if len(geometry) < 3:
            return 'Straight'
        total_angle = 0.0
        for i in range(1, len(geometry) - 1):
            p1 = geometry[i - 1]
            p2 = geometry[i]
            p3 = geometry[i + 1]
            v1 = (p1['lon'] - p2['lon'], p1['lat'] - p2['lat'])
            v2 = (p3['lon'] - p2['lon'], p3['lat'] - p2['lat'])
            angle = angle_between_vectors(v1, v2)
            total_angle += abs(angle)
        if total_angle < 10:
            return 'Straight'
        elif total_angle < 30:
            return 'Moderate Curve'
        else:
            return 'Sharp Curve'
    def angle_between_vectors(v1, v2):
        dot_prod = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if mag1 * mag2 == 0:
            return 0
        return math.degrees(math.acos(dot_prod / (mag1 * mag2)))
    curvature = calculate_curvature(geometry)
    return {
        'road_type': classify_road_type(road_type),
        'school_nearby': check_school_nearby(lat, lon),
        'traffic_density': traffic_estimate,
        'curvature': curvature
    }

def classify_road_type(road_type):
    if road_type in ['motorway', 'trunk', 'primary']:
        return 'Highway'
    elif road_type in ['secondary', 'tertiary']:
        return 'Urban Road'
    elif road_type in ['residential', 'service']:
        return 'Residential'
    else:
        return 'Unknown'

def check_school_nearby(lat, lon):
    query = f"""
    [out:json];
    node(around:500,{lat},{lon})[amenity=school];
    out;
    """
    response = requests.get("https://overpass-api.de/api/interpreter", params={'data': query})
    data = response.json()
    return len(data.get("elements", [])) > 0

def get_current_temperature(latitude, longitude):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}&current=temperature_2m,weathercode"
    )
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temperature = data['current']['temperature_2m']
        weather_code = data['current'].get('weathercode', 0)
        weather = classify_weather(weather_code, temperature)
        return temperature, weather
    else:
        print(f"Failed to get weather data. Status code: {response.status_code}")
        return None, "Unknown"

def classify_weather(weather_code, temp):
    if weather_code == 0 and temp >= 20:
        return "Clear"
    elif 45 <= weather_code <= 48:
        return "Foggy"
    elif weather_code >= 51:
        return "Rainy"
    else:
        return "Clear"

def get_time_period():
    now = datetime.now()
    hour = now.hour
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 20:
        return "Evening"
    else:
        return "Night"


# In[ ]:


# Sample location
latitude = 31.0469975
longitude = 31.3920175

# Fetch road, weather, and time data
road_info = get_road_data(latitude, longitude)
temperature, weather = get_current_temperature(latitude, longitude)
time_period = get_time_period()

# Output results
print("📍 Location Info:")
print(f"Coordinates: ({latitude}, {longitude})")
print(f"Time Period: {time_period}")
print(f"Temperature: {temperature}°C" if temperature is not None else "Temperature: unavailable")
print(f"Weather: {weather}")
print(f"Road Type: {road_info['road_type']}")
print(f"School Nearby: {road_info['school_nearby']}")
print(f"Traffic Density: {road_info['traffic_density']}")
print(f"Curvature: {road_info['curvature']}")


# In[4]:


get_ipython().system('apt-get install graphviz -y')
get_ipython().system('pip install graphviz')
from graphviz import Digraph
from IPython.display import Image

dot = Digraph(comment='DNN Model Architecture')
dot.attr(rankdir='TB', size='8')

dot.node('input', 'Input\n(input_dim=6)\nاسم: input')
dot.node('dense1', 'Dense (128)\nActivation: relu\nاسم: dense_1')
dot.node('drop1', 'Dropout (0.2)\nاسم: dropout_1')
dot.node('dense2', 'Dense (64)\nActivation: relu\nاسم: dense_2')
dot.node('drop2', 'Dropout (0.2)\nاسم: dropout_2')
dot.node('dense3', 'Dense (32)\nActivation: relu\nاسم: dense_3')
dot.node('output', 'Dense (1)\nOutput Layer\nاسم: output')

dot.edges([
    ('input', 'dense1'),
    ('dense1', 'drop1'),
    ('drop1', 'dense2'),
    ('dense2', 'drop2'),
    ('drop2', 'dense3'),
    ('dense3', 'output')
])

file_path = 'D:\project\project 2'
dot.render(file_path, format='png', cleanup=True)
Image(file_path+'.png')

