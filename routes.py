from flask import Flask, request, jsonify, render_template, redirect, url_for, redirect, session
import logging
from datetime import datetime, timedelta
from weather_service import fetch_weather_data, WeatherDataError
from analysis import analyze_weather_data
import os
from dotenv import load_dotenv
from model import preprocess_weather_data, validate_and_adjust_data, create_and_train_model, predict_future_weather, plot_weather_trend

# Variable global para almacenar las predicciones
predictions_data = []

app = Flask(__name__)

# Configuramos la clave secreta para manejar sesiones
app.secret_key = os.getenv("SECRET_KEY", "your_default_secret_key")

# Configuramos el nivel de logging
app.logger.setLevel(logging.INFO)

# Configuramos el formato y el manejador de logs que estaremos usando
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

# Cargamos las variables de entorno
load_dotenv()
API_KEY = os.getenv("WEATHER_API_KEY")

# Validamos que exista una clave de API para evitar errores
if not API_KEY:
    raise ValueError("API key not found.")

# Cantidad de días históricos que vamos a obtener de la API
FETCH_DAYS = 90

@app.route('/')
def index():
    """
    Ruta inicial que carga la página principal.
    """
    return render_template('index.html')

@app.route('/process_data', methods=['POST'])
def process_data():
    global predictions_data  # Declarar que se usará la variable global
    """
    Realiza la carga de los datos para la ciudad especificada por el usuario, los procesa y entrena un modelo al igual que realiza predicciones.
    """

    # Obtenemos la ciudad ingresada por el usuario
    city = request.form['city'].strip()
    if not city:
        return render_template('error.html', message="City name is required.")

    try:
        # Rango de fechas para obtener datos históricos
        end_date = datetime.now()
        start_date = end_date - timedelta(days=FETCH_DAYS)

        # Llamamos a la API para obtener datos históricos y actuales
        weather_data, current_weather = fetch_weather_data(city, start_date, end_date, API_KEY)

         # Preprocesamos los datos obtenidos
        X, y = preprocess_weather_data(weather_data)
        validate_and_adjust_data(X, y, app)

        # Creamos y entrenamos el modelo con los datos históricos
        model = create_and_train_model(weather_data, app)

        # Generamos predicciones para los próximos 5 días
        predictions = predict_future_weather(model, current_weather, num_days=5)
        predictions_data = [
            {"date": (datetime.now() + timedelta(days=i + 1)).strftime('%Y-%m-%d'), "predicted_temp": float(temp)}
            for i, temp in enumerate(predictions)
        ]
        app.logger.info(f"Predictions: {predictions_data}")

        # Guardamos el resumen del análisis y los datos actuales en la sesión
        session['summary'] = analyze_weather_data(weather_data)
        session['current_weather'] = current_weather
        session['city'] = city

        # Redirigimos a la página de resultados
        return redirect(url_for('results'))
    except WeatherDataError as e:
        # Si ocurre algún error al obtener datos o procesarlos, mandamos a la pantalla de error
        return render_template('error.html', message=str(e))

@app.route('/results')
def results():
    """
    Muestra los resultados del análisis y las predicciones.
    """
    # Recuperamos los datos de la sesión y los mostramos 
    predictions = session.get('predictions', [])
    summary = session.get('summary', {})
    current_weather = session.get('current_weather', {})
    city = session.get('city', "Unknown")

    # Si no hay predicciones disponibles, mostramos un mensaje de error
    if not predictions:
        return render_template('error.html', message="No predictions available. Please try again.")
    
    # Renderizamos la página de resultados con los datos obtenidos
    return render_template('results.html', city=city, summary=summary, current_weather=current_weather)


@app.route('/predictions')
def predictions():
    """
    Muestra las predicciones del clima para los próximos días.
    """
    global predictions_data  # Usamos la variable global para manejar predicciones
    if not predictions_data:
        return render_template('error.html', message="No hay predicciones disponibles. Realiza un análisis primero.")
    
    # Renderizamos la página de predicciones con las predicciones calculadas
    return render_template('predictions.html', predictions=predictions_data)