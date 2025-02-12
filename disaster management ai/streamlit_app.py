import streamlit as st
import requests
import folium
import xml.etree.ElementTree as ET
import requests

import plotly.express as px
from streamlit_folium import st_folium

# APIs for real-time data
GDACS_API_URL = "https://www.gdacs.org/xml/rss.xml"  # GDACS RSS feed for global disaster alerts

NEWS_API_KEY = "984e4e810426425eaaec223dc2195861"  # Replace with your News API key
WEATHER_API_KEY = "625f44b0c287216be043fb1a0175f5bd"  # Replace with your OpenWeatherMap API key

# Free AI Chatbot using Hugging Face API (No API key required)
def chat_with_huggingface(user_input):
    API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    headers = {"Authorization": "Bearer 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"}
    payload = {"inputs": user_input}
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error: Unable to get response. Status Code: {response.status_code}, Message: {response.text}"

# Function to get real-time disaster news
def get_disaster_news():
    url = f"https://newsapi.org/v2/everything?q=disaster&apiKey={NEWS_API_KEY}"
    response = requests.get(url).json()
    articles = response.get("articles", [])
    return articles[:5]  # Return top 5 articles

# Function to get real-time weather
def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url).json()
    if response.get("cod") == 200:
        return response
    return None

# Function to get real-time GDACS disaster data
def get_gdacs_disasters():
    response = requests.get(GDACS_API_URL)
    
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        events = []
        
        for item in root.findall(".//item"):
            title = item.find("title").text
            link = item.find("link").text
            description = item.find("description").text
            events.append({"title": title, "link": link, "description": description})

        return events
    else:
        return [{"error": f"GDACS RSS Fetch Error: {response.status_code}"}]

# Sidebar Navigation
st.sidebar.title("🌍 AI Disaster Dashboard By Akash Chaudhary")
page = st.sidebar.radio("Go to", ["Home", "Predict", "Live News & Weather", "Chatbot"])

# 🎯 Home Page
if page == "Home":
    st.title("🌍 AI Disaster Prediction Dashboard")
    st.write("Welcome to the AI-powered disaster impact predictor. Use the sidebar to navigate.")

    # 📊 Example Data for Disaster Occurrences
    df = px.data.gapminder().query("year == 2007")
    fig = px.scatter(df, x="gdpPercap", y="lifeExp", size="pop", color="continent",
                     hover_name="country", log_x=True, size_max=60)
    st.plotly_chart(fig)

# 🚀 Prediction Page
elif page == "Predict":
    st.title("🧠 Predict Disaster Impact")

    magnitude = st.number_input("Magnitude", min_value=0.0, value=5.0)
    population_density = st.number_input("Population Density (people per sq km)", min_value=0, value=1000)
    year = st.number_input("Year", min_value=1900, max_value=2100, value=2023)
    
    disaster_type = st.selectbox("Disaster Type", ["Flood", "Earthquake", "Storm", "Landslide", "Drought"])
    disaster_code = {"Flood": 0, "Earthquake": 1, "Storm": 2, "Landslide": 3, "Drought": 4}[disaster_type]

    if st.button("Predict Damage Cost"):
        with st.spinner("🔄 Processing... Please wait"):
            try:
                url = "http://127.0.0.1:5000/predict"
                data = {
                    "magnitude": magnitude,
                    "population_density": population_density,
                    "year": year,
                    "disastertype_code": disaster_code
                }

                response = requests.post(url, json=data, timeout=10)  # Add timeout
                
                # Handle API response
                if response.status_code == 200:
                    prediction = response.json().get("predicted_damage_cost", "Error")
                    st.success(f"💰 Predicted Damage Cost: ${prediction:,.2f}")
                    
                    # 🌍 Show Disaster Location on Map
                    m = folium.Map(location=[0, 0], zoom_start=2)  # Default location
                    folium.Marker(
                        location=[0, 0],
                        popup=f"Predicted Cost: ${prediction:,.2f}",
                        icon=folium.Icon(color="red")
                    ).add_to(m)
                    st_folium(m)  # Only call this once

                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.json().get('error', 'Unknown error')}")
        
            except requests.exceptions.ConnectionError:
                st.error("🚨 Connection Error: Ensure Flask is running (`python main.py`)")
            except requests.exceptions.Timeout:
                st.error("⏳ Request Timed Out: The server took too long to respond.")
            except Exception as e:
                st.error(f"⚠️ Unexpected Error: {str(e)}")

    # 🌦 Fetch Weather Data
    city = st.text_input("Enter City for Weather Info", "Los Angeles")
    weather_data = get_weather(city)
    if weather_data:
        st.write(f"🌦 **Weather in {city}:** {weather_data['weather'][0]['description'].capitalize()}")
        st.write(f"🌡 **Temperature:** {weather_data['main']['temp']}°C")
        st.write(f"💨 **Wind Speed:** {weather_data['wind']['speed']} m/s")

# 📰 Live Disaster News & Weather
elif page == "Live News & Weather":
    st.title("📰 Live Disaster News & Weather")
    
    # Fetch & Display News
    st.subheader("📢 Latest Disaster News")
    news_articles = get_disaster_news()
    for article in news_articles:
        st.markdown(f"🔹 **[{article['title']}]({article['url']})**")
        st.write(f"📅 {article['publishedAt']}")
        st.write(f"📝 {article['description']}")
        st.image(article['urlToImage'], width=500)

    # Fetch GDACS Disaster Data
    st.subheader("📊 Global Disaster Alerts")
    gdacs_data = get_gdacs_disasters()
    for event in gdacs_data:
        if "error" not in event:
            st.markdown(f"🔹 **[{event['title']}]({event['link']})**")
            st.write(f"📝 {event['description']}")
        else:
            st.error(event["error"])

# 🤖 Chatbot Page
elif page == "Chatbot":
    st.title("🤖 AI Chatbot for Emergency Assistance")
    user_query = st.text_input("Ask a disaster-related question:")
    
    if st.button("Ask AI"):
        if user_query:
            reply = chat_with_huggingface(user_query)
            st.write("💬 AI Response: ", reply)
        else:
            st.warning("Please enter a question.")
