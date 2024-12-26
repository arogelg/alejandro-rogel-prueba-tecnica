# Este archivo configura el contenedor Docker para ejecutar la aplicación.

# Usamos una imagen oficial de Python como base
FROM python:3.9-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos todos los archivos del proyecto actual al contenedor
COPY . /app

# Instalamos las bibliotecas necesarias para la aplicación desde el archivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Exponemos el puerto 5000, que es donde la aplicación estará escuchando
EXPOSE 5000

# Definimos el comando predeterminado para iniciar la aplicación
CMD ["python", "app.py"]
