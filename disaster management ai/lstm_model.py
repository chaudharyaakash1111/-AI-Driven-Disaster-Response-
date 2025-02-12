import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv("GDIS_disasterlocations.csv")

# Data Preprocessing
df_cleaned = df.dropna(subset=['latitude', 'longitude', 'casualties'])
df_cleaned['disastertype_code'] = df_cleaned['disastertype'].astype('category').cat.codes

# Selecting Features & Target
features = ['latitude', 'longitude', 'disastertype_code']
target = 'casualties'

# Normalize Data
scaler = MinMaxScaler()
df_cleaned[features] = scaler.fit_transform(df_cleaned[features])

# Prepare Data for LSTM
X = np.array(df_cleaned[features]).reshape(-1, len(features), 1)
y = np.array(df_cleaned[target])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM Model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(len(features), 1)),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

# Save Model
model.save("disaster_lstm_model.h5")
joblib.dump(scaler, "scaler.pkl")
