import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify
import streamlit as st
import joblib

# Step 1: Data Collection (Scraping Disaster Data)
def fetch_disaster_data():
    url = "https://www.emdat.be/"
    response = requests.get(url)
    if response.status_code == 200:
        # Simulating dataset (Replace with actual scraping/parsing logic)
        data = {
            "year": np.arange(2000, 2025),
            "earthquakes": np.random.randint(1, 20, 25),
            "floods": np.random.randint(10, 50, 25),
            "hurricanes": np.random.randint(0, 10, 25),
            "casualties": np.random.randint(100, 5000, 25)
        }
        return pd.DataFrame(data)
    else:
        print("Error fetching data")
        return None

# Step 2: Preprocessing & Feature Engineering
def preprocess_data(df):
    df['total_disasters'] = df['earthquakes'] + df['floods'] + df['hurricanes']
    df['casualties'] = np.log1p(df['casualties'])  # Log transform for normalization
    return df

# Step 3: Build LSTM Model
def build_lstm_model():
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(3, 1)),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Step 4: Train the Model
def train_model(df):
    X = df[['earthquakes', 'floods', 'hurricanes']].values.reshape(-1, 3, 1)
    y = df['casualties'].values
    model = build_lstm_model()
    model.fit(X, y, epochs=50, batch_size=8, verbose=1)
    model.save("disaster_model.h5")
    joblib.dump(df, "disaster_data.pkl")  # Save processed data
    return model

# Step 5: Flask API for Predictions
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model = tf.keras.models.load_model("disaster_model.h5")
    X_input = np.array([[data['earthquakes'], data['floods'], data['hurricanes']]]).reshape(-1, 3, 1)
    prediction = model.predict(X_input)
    return jsonify({"predicted_casualties": float(np.expm1(prediction[0][0]))})

# Streamlit logic removed as it is handled in streamlit_app.py
