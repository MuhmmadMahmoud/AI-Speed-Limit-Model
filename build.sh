#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Create model and scaler
python3 -c "
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the original model to get weights
original_model = load_model('speed_limit_model.h5')
weights = original_model.get_weights()

# Create the model architecture
model = Sequential([
    Dense(128, input_dim=6, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1)
])

# Set the weights from the original model
model.set_weights(weights)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Create and save the scaler
scaler = MinMaxScaler()
# Initialize with typical ranges for our features
scaler.fit(np.array([
    [0, 0, 0, 0, 0, 0],  # min values
    [2, 1, 1, 2, 1, 3]   # max values
]))

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the model
model.save('speed_limit_model_saved', save_format='tf')
print('Model and scaler created and saved successfully!')
" 