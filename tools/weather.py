import requests
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a specific location."""
    
    # 1. Geocoding: Get coordinates for the location
    geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
    try:
        geo_response = requests.get(geocoding_url).json()
        if not geo_response.get("results"):
            return f"Couldn't find coordinates for '{location}'."
        
        result = geo_response["results"][0]
        lat = result["latitude"]
        lon = result["longitude"]
        city = result.get("name", location)
        country = result.get("country", "")
        
        # 2. Weather: Get current weather using coordinates
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_response = requests.get(weather_url).json()
        
        if "current_weather" not in weather_response:
            return f"Couldn't retrieve weather data for {city}."
        
        current = weather_response["current_weather"]
        temp = current["temperature"]
        windspeed = current["windspeed"]
        
        return f"Current weather in {city}, {country}: {temp}°C, Windspeed: {windspeed} km/h."
        
    except Exception as e:
        return f"Error fetching weather: {str(e)}"
