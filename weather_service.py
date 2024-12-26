import requests
from datetime import timedelta

class WeatherDataError(Exception):
    """
    Clase especial para errores, la usamos para manejar casos específicos donde la API falla o devuelve datos inválidos.
    """
    pass

def fetch_weather_data(city, start_date, end_date, api_key):
    """
    Obtiene datos meteorológicos para una ciudad específica y un rango de fechas utilizando WeatherAPI.

    :param city: Nombre de la ciudad para la cual obtenemos los datos del clima.
    :param start_date: Fecha de inicio para los datos históricos.
    :param end_date: Fecha de fin para los datos históricos.
    :param api_key: Clave API para acceder a WeatherAPI.
    :return: Una tupla que contiene:
        - Lista de diccionarios con todos los datos meteorológicos para cada día en el rango.
        - Diccionario con los datos meteorológicos del último día del rango.
    :raises WeatherDataError: Si la solicitud a la API falla o no se devuelve ningún dato.
    """
    current_date = start_date  # Inicializamos con la fecha de inicio del rango.
    weather_data = []  # Lista para almacenar los datos meteorológicos de cada día.

    # Iteramos sobre el rango de fechas, día por día
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')  # Formateamos la fecha actual como YYYY-MM-DD.
        weather_url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date_str}"  # Endpoint de la API para la fecha especificada.
        
        try:
            # Realizamos una solicitud GET a la API WeatherAPI
            response = requests.get(weather_url, timeout=5)
        except requests.exceptions.RequestException as e:
            # Manejo de errores de conexión o tiempo de espera
            raise WeatherDataError(f"Error al conectar con WeatherAPI: {e}")
        
        # Verificamos si la respuesta fue exitosa
        if response.status_code == 200:
            forecast = response.json().get("forecast", {}).get("forecastday", [])[0]  # Extraemos los datos de pronóstico para el día específico.
            if forecast:
                day = forecast.get("day", {})
                # Agregamos los datos relevantes del clima a la lista
                weather_data.append({
                    "date": forecast.get("date"),
                    "average_temp": day.get("avgtemp_c"),
                    "max_temp": day.get("maxtemp_c"),
                    "min_temp": day.get("mintemp_c"),
                    "precipitation": day.get("totalprecip_mm"),
                    "humidity": day.get("avghumidity"),
                    "condition": day.get("condition", {}).get("text"),
                })
        else:
            # Lanzamos un error personalizado si la respuesta de la API no es exitosa
            raise WeatherDataError(f"Error al obtener datos meteorológicos para {city}: {response.status_code} - {response.text}")

        # Avanzamos al siguiente día en el rango
        current_date += timedelta(days=1)

    # Lanzamos un error si no se obtuvieron correctamente los datos meteorológicos
    if not weather_data:
        raise WeatherDataError(f"No hay datos meteorológicos disponibles para {city}.")

    # Devolvemos los datos del clima y los datos del día más reciente
    return weather_data, weather_data[-1]