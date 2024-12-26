import pandas as pd

def analyze_weather_data(weather_data):
    """
    Analiza los datos meteorológicos y genera un resumen.
    
    :param weather_data: Lista de diccionarios con datos históricos del clima.
    :return: Diccionario con el resumen del análisis.
    """
    # Convertimos los datos meteorológicos en un DataFrame de pandas
    df = pd.DataFrame(weather_data)

    # Generamos un resumen de los datos analizados
    summary = {
        "average_temp_overall": df['average_temp'].mean(),  # Temperatura promedio general
        "rainy_days": df[df['precipitation'] > 0].shape[0],  # Número de días con lluvia
        "total_precipitation": df['precipitation'].sum(),  # Precipitación total en el período analizado
        "max_temp": df['max_temp'].max(),  # Temperatura máxima registrada
        "min_temp": df['min_temp'].min()  # Temperatura mínima registrada
    }

    return summary