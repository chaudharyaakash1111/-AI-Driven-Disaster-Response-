import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify

import joblib

# Load disaster dataset (example CSV)
df = pd.read_csv("disaster_data.csv")

# Data Preprocessing
df['date'] = pd.to_datetime(df['date'])  # Convert date column to datetime
df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill NaN values for numeric columns only

# Feature Engineering
df['year'] = pd.to_datetime(df['date']).dt.year
X = df[['magnitude', 'population_density', 'year']]  # Features for the model
y = df['damage_cost']

# Load the trained model
if os.path.exists("trained_model.pkl"):
    model = joblib.load("trained_model.pkl")
else:
    # Model Training
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "trained_model.pkl")







# Data Collection
url = "https://www.emdat.be/"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
data = []  # Parse the website to extract disaster-related data
# Example parsing logic to extract disaster data
for item in soup.find_all('div', class_='disaster-item'):
    date = item.find('span', class_='date').text
    magnitude = float(item.find('span', class_='magnitude').text)
    population_density = float(item.find('span', class_='population-density').text)
    damage_cost = float(item.find('span', class_='damage-cost').text)
    data.append({'date': date, 'magnitude': magnitude, 'population_density': population_density, 'damage_cost': damage_cost})

# Load disaster dataset (example CSV)
df = pd.read_csv("disaster_data.csv")

# Data Preprocessing
df['date'] = pd.to_datetime(df['date'])  # Convert date column to datetime
df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill NaN values for numeric columns only

# Feature Engineering
df['year'] = pd.to_datetime(df['date']).dt.year
X = df[['magnitude', 'population_density', 'year']]  # Features for the model
y = df['damage_cost']

# Train-Test Split
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "trained_model.pkl")

# Predictions
# Predictions
y_pred = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"MAE: {mae}, MSE: {mse}")

# Flask API
app = Flask(__name__)

@app.route('/', methods=['GET'])  # Add this to show a home page
def home():
    return """
    Welcome to the AI Disaster Prediction API! 
    Use /predict for predictions.
    Available endpoints:
    - POST /predict: Predict disaster casualties based on input data.
    """


@app.route('/predict', methods=['POST'])

def predict():
    try:
        data = request.get_json()
        features = pd.DataFrame([[data['magnitude'], data['population_density'], data['year']]], columns=['magnitude', 'population_density', 'year'])

        print("Features for prediction:", features)  # Debugging statement
        prediction = model.predict(features)
        print("Prediction result:", prediction)  # Debugging statement

        return jsonify({'predicted_damage_cost': prediction[0]})
    except KeyError as e:
        return jsonify({'error': f'Missing key: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
