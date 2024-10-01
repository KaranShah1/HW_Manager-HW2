import streamlit as st
from openai import OpenAI
import requests

# Set API Keys from Streamlit secrets
client = OpenAI(api_key=st.secrets["openai"])

weather_api_key = st.secrets["weather"]  # Accessing OpenWeatherMap API key from Streamlit secrets


def get_weather(location="Syracuse, NY"):
    """Fetch weather data from OpenWeatherMap."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={weather_api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "location": data["name"],
            "temperature": round(data["main"]["temp"], 2),
            "feels_like": round(data["main"]["feels_like"], 2),
            "temp_min": round(data["main"]["temp_min"], 2),
            "temp_max": round(data["main"]["temp_max"], 2),
            "humidity": round(data["main"]["humidity"], 2),
            "weather": data["weather"][0]["description"]
        }
    else:
        return {"error": "Could not retrieve weather data."}

def get_clothing_suggestions(weather_info):
    """Function to call OpenAI's API for clothing suggestions based on the weather."""
    prompt = (f"The current weather in {weather_info['location']} is {weather_info['temperature']}°C, "
              f"with a 'feels like' temperature of {weather_info['feels_like']}°C. "
              f"The humidity is {weather_info['humidity']}%. "
              f"Based on this, what kind of clothing would you suggest for someone traveling today?")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Specify model as per new API
        messages=[
            {"role": "system", "content": "You are a helpful assistant that gives weather-based clothing advice in one line. Provide clothing suggestions and advice on whether it’s a good day for a picnic."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    # Extract the text from the completion response
    return response.choices[0].message.content.strip()

def llm_tool(location):
    """Function to get weather details."""
    if not location:
        location = "Syracuse, NY"  # Default location
    
    weather_data = get_weather(location)
    
    if 'error' in weather_data:
        return f"Error: {weather_data['error']}"
    
    return (f"The current weather in {weather_data['location']}:\n"
            f"Temperature: {weather_data['temperature']}°C\n"
            f"Feels Like: {weather_data['feels_like']}°C\n"
            f"Min Temp: {weather_data['temp_min']}°C\n"
            f"Max Temp: {weather_data['temp_max']}°C\n"
            f"Humidity: {weather_data['humidity']}%\n"
            f"Conditions: {weather_data['weather']}")

# Streamlit UI
st.title("Weather and Clothing Suggestion Bot")

# Get user input for location
user_input = st.text_input("Enter a city (leave blank for default - Syracuse, NY):")

# Create two buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("Get Weather Suggestion", key="key1"):
        suggestion = llm_tool(user_input)
        st.write(suggestion)

with col2:
    if st.button("Get Clothing Suggestion", key="key2"):
        weather_data = get_weather(user_input)
        if 'error' not in weather_data:
            clothing_suggestion = get_clothing_suggestions(weather_data)
            st.write(f"Weather Info:\n{llm_tool(user_input)}")
            st.write(f"\nClothing Suggestion: {clothing_suggestion}")
        else:
            st.write(f"Error: {weather_data['error']}")
