import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta


def preprocess_weather_data(weather_data):
    """
    Esta funcion preprocesa los datos meteorológicos para generar características útiles.
    :param weather_data: Lista de diccionarios con datos históricos del clima.
    :return: DataFrame preprocesado con X (features) y y (target).
    """
    # Convertimos los datos en un DataFrame de pandas
    df = pd.DataFrame(weather_data)
    
    # Generamos características de series temporales que usaremos para nuestro modelo
    df['rolling_avg_temp'] = df['average_temp'].rolling(window=3, min_periods=1).mean()  # Promedio de temperatura móvil (3 días)
    df['rolling_max_temp'] = df['max_temp'].rolling(window=3, min_periods=1).max()  # Máximo de temperatura móvil (3 días)
    df['rolling_min_temp'] = df['min_temp'].rolling(window=3, min_periods=1).min()  # Mínimo de temperatura móvil (3 días)
    df['cumulative_precipitation'] = df['precipitation'].cumsum()  # Precipitación acumulada

    # Droppeamos condition ya que muchas veces no viene nada en este campo
    df = df.drop(columns=['condition'], errors='ignore')
    
    # Separamos características (X) y objetivo (y)
    X = df.drop(columns=['average_temp', 'date'])  # Características (sin la columna objetivo)
    y = df['average_temp']  # Objetivo (temperatura promedio)
    
    return X, y


def validate_and_adjust_data(X, y, app):
    """
    Esta funcion valida el conjunto de datos preprocesados y sugiere o aplica ajustes básicos.
    :param X: DataFrame de características.
    :param y: Serie u objetivo que queremos predecir.
    :return: DataFrame ajustado (X) y objetivo ajustado (y).
    """

    app.logger.info("=== INICIO: Validación de los datos preprocesados ===")

    # Verificamos valores faltantes
    missing_values = X.isnull().sum()
    total_missing = missing_values.sum()

    app.logger.info(f"Valores faltantes totales: {total_missing}")
    if missing_values.sum() > 0:
        app.logger.info("Hay valores faltantes, reemplazando con la mediana...")
        X = X.fillna(X.median())  #Hacemos un remplazo de los valores faltantes con la mediana

    # Resumen estadístico de las características de nuestros datos
    app.logger.info("Promedio de características:")
    app.logger.info(X.mean())


    app.logger.info("Rango de temperaturas:")
    app.logger.info(f"Max Temp: {X['max_temp'].max()}, Min Temp: {X['min_temp'].min()}")

    # Calculamos la estadisticas descriptivas de nuestro objetivo
    app.logger.info("Resumen de temperaturas promedio (objetivo):")
    app.logger.info(f"Media: {y.mean()}, Máxima: {y.max()}, Mínima: {y.min()}")

    #  Detectamos valores extremos en el objetivo
    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = y[(y < lower_bound) | (y > upper_bound)]
    app.logger.info(f"Valores extremos detectados en el objetivo: {len(outliers)}")
    if len(outliers) > 0:
        app.logger.info("Aplicando truncado para manejar valores extremos ")
        y = np.clip(y, lower_bound, upper_bound)

    # Obtenemos y analizamos las correlaciones
    df = X.copy()
    df['target'] = y
    correlations = df.corr()['target'].sort_values(ascending=False)
    app.logger.info("Principales correlaciones con el objetivo:")
    app.logger.info(correlations.head(5))  # Mostramos las 5 más relevante

    # Eliminamos las características con baja correlación
    low_corr_features = correlations[correlations.abs() < 0.1].index
    if len(low_corr_features) > 0:

        app.logger.info(f"Eliminando características con baja correlación: {list(low_corr_features)}")
        X = X.drop(columns=low_corr_features, errors='ignore')
    

    app.logger.info("=== FIN: Validación y ajustes completados ===")
    return X, y

def evaluate_model(model, predictions, y_true):
    """
    Evalúa un modelo de regresión utilizando métricas comunes.
    :param model: Modelo de regresión entrenado.
    :param predictions: Predicciones generadas por el modelo.
    :param y_true: Valores reales (verdaderos).
    :return: Diccionario con métricas de evaluación.
    """
    # Calculamos el error cuadrático medio (MSE) para medir el promedio de los errores al cuadrado
    mse = mean_squared_error(y_true, predictions)

    # Calculamos el error absoluto medio (MAE) para medir el promedio absoluto de los errores
    mae = mean_absolute_error(y_true, predictions)

    # Calculamos el coeficiente de determinación (R²) para evaluar qué tan bien el modelo explica la varianza
    r2 = r2_score(y_true, predictions)

    # Devolvemos las métricas como un diccionario
    return {
        "mean_squared_error": mse,
        "mean_absolute_error": mae,
        "r2_score": r2
    }



def create_and_train_model(data, app):
    """
    Crear y entrenar un modelo de regresión con los datos proporcionados.
    :param data: Lista de diccionarios con datos meteorológicos.
    """
    # Preprocesamos los datos
    X, y = preprocess_weather_data(data)
    
    # Validamos los datos preprocesados
    validate_and_adjust_data(X, y, app)

    #Dividimos los datos en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenamos el modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Probamos el modelo con el conjunto de entrenamiento
    predictions = model.predict(X_test)
    print(f"Predicciones: {predictions}")
    print(f"Valores reales: {y_test}")
    
    # Calculamos métricas para evaluar el modelo llamando a nuestra funcion para esto
    metrics = evaluate_model(model, predictions, y_test)
    print(f"Resultados de la evaluación: {metrics}")

    return model


def predict_future_weather(model, last_known_data, num_days=3):
    """
    Predice el clima para los próximos días utilizando un modelo entrenado.
    Asegura que las características coincidan exactamente con las del modelo entrenado,
    y maneja columnas faltantes inicializándolas adecuadamente.
    :param model: Modelo entrenado.
    :param last_known_data: Diccionario con los datos más recientes.
    :param num_days: Número de días a predecir.
    :return: Lista de predicciones para los próximos días.
    """
  # Proceso inicial del diccionario para conservar solo las características necesarias
    processed_data = {
        'max_temp': last_known_data['max_temp'],
        'min_temp': last_known_data['min_temp'],
        'precipitation': last_known_data['precipitation'],
        'humidity': last_known_data['humidity'],
    }
    
    # Convertimos el diccionario procesado en un DataFrame
    current_data = pd.DataFrame([processed_data])
    
    # Inicializamos características adicionales necesarias para el modelo
    current_data['rolling_avg_temp'] = last_known_data.get('average_temp', 0)
    current_data['rolling_max_temp'] = last_known_data['max_temp']
    current_data['rolling_min_temp'] = last_known_data['min_temp']
    current_data['cumulative_precipitation'] = last_known_data['precipitation']
    
    # Reemplazamos NaN iniciales por precaución
    current_data.fillna(0, inplace=True)
    
    # Aseguramos que todas las columnas requeridas por el modelo estén presentes
    expected_features = [
        'max_temp', 'min_temp', 'precipitation', 'humidity',
        'rolling_avg_temp', 'rolling_max_temp', 'rolling_min_temp',
        'cumulative_precipitation'
    ]
    for feature in expected_features:
        if feature not in current_data:
            current_data[feature] = 0  # Inicializamos con 0 si falta la característica
    
    # Filtramos las columnas para que coincidan exactamente con las del modelo
    current_data = current_data[expected_features]
    
    future_predictions = []  # Almacenará las predicciones futuras
    
    for day in range(num_days):
        # Convertimos la fila actual en un DataFrame para el modelo
        input_features = pd.DataFrame([current_data.iloc[0].values], columns=expected_features)
        
        # Generamos la predicción
        prediction = model.predict(input_features)[0]
        future_predictions.append(prediction)

        # Introducimos variación aleatoria para simular incertidumbre que se presenta en el clima
        prediction += np.random.uniform(-1, 1)  
        
        # Actualizamos características dinámicas para el siguiente día
        current_data.at[0, 'rolling_avg_temp'] = (
            current_data.at[0, 'rolling_avg_temp'] * 2 + prediction
        ) / 3  
        current_data.at[0, 'rolling_max_temp'] = max(
            current_data.at[0, 'rolling_max_temp'], prediction
        )  
        current_data.at[0, 'rolling_min_temp'] = min(
            current_data.at[0, 'rolling_min_temp'], prediction
        )  
        current_data.at[0, 'cumulative_precipitation'] += current_data.at[0, 'precipitation']
        
        # Nos aseguramos de que no haya NaN después de actualizar
        current_data.fillna(0, inplace=True)
    
    return future_predictions

def plot_weather_trend(historical_data, predictions):
    """
    Genera un gráfico que combina los datos históricos con las predicciones futuras.
    :param historical_data: Lista de diccionarios con datos históricos (cada elemento incluye 'date' y 'average_temp').
    :param predictions: Lista de diccionarios con predicciones futuras (cada elemento incluye 'date' y 'predicted_temp').
    """
    # Convertimos los datos históricos en listas para graficar
    historical_dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in historical_data]
    historical_temps = [d['average_temp'] for d in historical_data]

    # Convertimos nuestras predicciones en listas para graficar
    try:
        prediction_dates = [
            datetime.strptime(p['date'], '%Y-%m-%d') for p in predictions if 'date' in p
        ]
    except KeyError as e:
        raise ValueError(f"Elemento en 'predictions' no contiene la clave 'date': {e}")
    predicted_temps = [p['predicted_temp'] for p in predictions]

    # Creamos la figura y los ejes
    plt.figure(figsize=(10, 6))

    # Graficamos los datos históricos
    plt.plot(historical_dates, historical_temps, label='Datos Históricos', marker='o', linestyle='-', color='blue')

    # Graficamos las predicciones futuras
    plt.plot(prediction_dates, predicted_temps, label='Predicciones Futuras', marker='o', linestyle='--', color='orange')

    # Formateamos el gráfico
    plt.title('Tendencia del Clima: Datos Históricos y Predicciones')
    plt.xlabel('Fecha')
    plt.ylabel('Temperatura (°C)')
    plt.grid(True)
    plt.legend()

    # Finalmente mostramos el gráfico y lo guardamos
    plt.tight_layout()
    plt.savefig('static/weather_trend.png')  
    plt.close()