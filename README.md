# Predicción del Clima y Análisis de Datos

Este proyecto es una aplicación web desarrollada con Flask que permite analizar datos históricos del clima, entrenar un modelo predictivo y generar predicciones del clima para los próximos días.

## Características Principales

- **Extracción de datos históricos del clima**: Utiliza WeatherAPI para obtener datos meteorológicos históricos de una ciudad específica.
- **Preprocesamiento de datos**: Convierte y valida los datos para que sean utilizables en un modelo de machine learning.
- **Modelado predictivo**: Entrena un modelo de regresión basado en un `RandomForestRegressor` para predecir temperaturas futuras.
- **Interfaz amigable**: Proporciona una interfaz web simple para visualizar análisis y predicciones.

## Requisitos

- **Python 3.9 o superior**
- **Docker** (opcional, para ejecutar en un contenedor)
- **Librerías necesarias**: Las dependencias están incluidas en el archivo `requirements.txt`. Entre ellas se encuentran:

  - Flask
  - requests
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - python-dotenv

## Instalación y Ejecución

### 1. Clona el Repositorio

```bash
git clone https://github.com/tu_usuario/tu_repositorio.git
cd tu_repositorio
```

### 2. Configura tus Variables de Entorno

Crea un archivo `.env` en el directorio raíz del proyecto y agrega lo siguiente:

```makefile
WEATHER_API_KEY=tu_clave_de_weather_api
SECRET_KEY=tu_clave_secreta_para_sesiones
```

### 3. Instala las Dependencias
Si no usas Docker, instala las dependencias manualmente:

```
bash
Copy code
pip install -r requirements.txt
```

### 4. Ejecuta la Aplicación
Usando Python:
```
bash
Copy code
python app.py
```
La aplicación estará disponible en: http://127.0.0.1:5000

Usando Docker:

```
bash
Copy code
docker build -t weather-app .
docker run -p 5000:5000 weather-app
```

### Uso

Accede a la aplicación en http://127.0.0.1:5000.
Ingresa el nombre de una ciudad (en inglés) y haz clic en "Analizar Clima".
Visualiza el análisis en la página de resultados, que incluye:
Un resumen estadístico.
Un gráfico de tendencias del clima.
Ve a la página "Predicciones" para revisar las predicciones generadas para los próximos días.

#### Estructura del Proyecto

```
.
├── app.py                # Archivo principal para iniciar la aplicación
├── weather_service.py    # Funciones para obtener datos de WeatherAPI
├── analysis.py           # Funciones para analizar datos meteorológicos
├── model.py              # Código para preprocesamiento, modelado y predicción
├── static/
│   └── styles.css        # Archivos estáticos (estilos, imágenes, etc.)
├── templates/
│   ├── index.html        # Página de inicio
│   ├── results.html      # Página de resultados
│   └── predictions.html  # Página de predicciones
├── requirements.txt      # Dependencias del proyecto
├── Dockerfile            # Configuración para el contenedor Docker
└── README.md             # Instrucciones del proyecto
```


