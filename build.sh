#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Convert the model to SavedModel format
python3 -c "
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
model = load_model('speed_limit_model.h5')
save_model(model, 'speed_limit_model_saved', save_format='tf')
print('Model converted successfully!')
" 