Guía de Despliegue: Sistema de Predicción de Pacientes con Interfaz Web. Este documento proporciona una guía completa para configurar, ejecutar y desplegar el sistema de predicción de mortalidad en pacientes, incluyendo el backend de Flask y la interfaz de usuario web.📋 1. Estructura del ProyectoLa estructura de archivos recomendada para el proyecto completo es la siguiente:proyecto/
│
├── 📂 strats_data/
│   ├── metadata.pkl             # Metadatos (scalers, feature maps)
│   ├── train_data.pkl           # Datos de entrenamiento preprocesados
│   ├── val_data.pkl             # Datos de validación preprocesados
│   └── test_data.pkl            # Datos de prueba preprocesados
│
├── 📂 results/
│   └── istrats_pretrained/      # Ejemplo de carpeta de un modelo
│       └── best_model.pth       # Pesos del modelo entrenado
│
├── 📜 1488_strats.csv            # Dataset original (si aplica)
├── 📜 strats_create_data.py      # Script para generar tripletas
├── 📜 strats_data_adapter.py     # Script para procesar y adaptar datos
├── 📜 models.py                  # Definición de las arquitecturas (STraTS, iSTraTS)
├── 📜 train.py                   # Pipeline de entrenamiento y evaluación
├── 📜 pretrain.py                # (Opcional) Pipeline de pre-entrenamiento
├── 📜 run_hyperparameter_search.py # Script para búsqueda de hiperparámetros
├── 📜 run_interpretability.py    # Lógica de interpretabilidad con SHAP
├── 📜 istrats_contribution.py    # Lógica de Contribution Scores para iSTraTS
│
├── 🌐 patient_prediction_webapp.html # Interfaz de usuario web (Frontend)
├── 🚀 flask_backend.py           # Servidor de la aplicación (Backend)
│
├── 📄 strats_config.json         # Configuración de hiperparámetros del modelo
├── 📄 requirements.txt           # Dependencias de Python
└── 📄 deployment_guide.md        # Esta guía
🚀 2. Instalación y Configuración2.1. DependenciasAsegúrate de tener Python 3.8+ instalado. Luego, instala todas las dependencias necesarias ejecutando:pip install -r requirements.txt
El contenido del archivo requirements.txt debe ser el siguiente para cubrir todo el proyecto (entrenamiento y despliegue):# Core ML & Data
torch
numpy
pandas
scikit-learn
shap
seaborn
matplotlib

# Web Backend
Flask
Flask-CORS

# Opcional para leer archivos excel si se usan en el futuro
openpyxl
2.2. Archivos del Modelo EntrenadoPara que la aplicación web funcione, necesitas tener los artefactos del modelo generados durante la Etapa 4: Entrenamiento y Evaluación Final. Asegúrate de que las siguientes carpetas y archivos existan:Carpeta de datos: strats_data/ con el archivo metadata.pkl.Carpeta de resultados: results/ con la subcarpeta del modelo que quieres usar (ej. results/istrats_pretrained/) conteniendo el archivo best_model.pth.⚙️ 3. Componentes de la Aplicación WebLa aplicación consta de dos partes principales que trabajan juntas.3.1. Backend (Flask)Archivo: flask_backend.pyFunción Principal:Inicia un servidor web usando Flask.Carga dinámica del modelo: Al arrancar, lee la variable de entorno MODEL_TO_LOAD para saber qué modelo cargar desde la carpeta results/. Por defecto, carga strats_pretrained.Endpoint /: Sirve la página principal patient_prediction_webapp.html.Endpoint /predict (POST): Recibe los datos del paciente desde la interfaz web en formato JSON.Preprocesamiento: Utiliza la función preprocess_new_patient para convertir los datos del formulario en los tensores (X, times, mask) que el modelo PyTorch necesita, usando los scalers y feature_maps guardados en metadata.pkl.Predicción: Pasa los tensores al modelo cargado y obtiene las probabilidades de salida.Respuesta: Devuelve la predicción (Evento/No Evento) y la probabilidad en formato JSON a la interfaz web.3.2. Frontend (HTML con JavaScript)Archivo: patient_prediction_webapp.htmlFunción Principal:Interfaz de Usuario: Presenta un formulario web estructurado para que el usuario ingrese los datos demográficos, comorbilidades y la evolución temporal del paciente.Recolección de Datos: A través de JavaScript, la función collectPatientData recopila todos los datos del formulario cuando se presiona el botón "Realizar Predicción".Formato de Tripletas: Transforma los datos del formulario en una lista de tripletas [tiempo, parametro, valor], que es el formato que el backend espera.Llamada a la API: Realiza una petición POST asíncrona al endpoint /predict del backend, enviando los datos del paciente en formato JSON.Visualización de Resultados: La función displayResult recibe la respuesta del backend y actualiza la página dinámicamente para mostrar el resultado de la predicción y una barra de probabilidad.🌐 4. Ejecución y Despliegue Local4.1. Ejecutar la AplicaciónPara iniciar la aplicación web, simplemente ejecuta el script del backend.# Para usar el modelo por defecto ('strats_pretrained')
python flask_backend.py

# Para especificar un modelo diferente (ej: 'istrats_scratch')
MODEL_TO_LOAD=istrats_scratch python flask_backend.py
Una vez ejecutado, el servidor estará disponible y podrás acceder a la aplicación abriendo la siguiente URL en tu navegador web:➡️ http://localhost:50004.2. Probar la API (Opcional)Puedes probar el endpoint de predicción directamente usando herramientas como curl:curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": [
      [0.0, "Sexo", 1.0],
      [0.0, "Edad", 75.0],
      [0.0, "micro5", 0.0],
      [0.0, "Insuficiencia cardiaca", 1.0],
      [1.0, "Insuficiencia cardiaca", 1.0]
    ]
  }'
🐳 5. Despliegue con Docker (Avanzado)Para un despliegue más robusto y portable, puedes usar Docker.5.1. DockerfileCrea un archivo llamado Dockerfile en la raíz del proyecto:# Usar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de dependencias e instalarlas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todos los archivos del proyecto al contenedor
COPY . .

# Exponer el puerto en el que corre la aplicación
EXPOSE 5000

# Comando para ejecutar la aplicación al iniciar el contenedor
CMD ["python", "flask_backend.py"]
5.2. Comandos de DockerConstruir la imagen de Docker:docker build -t mi-app-prediccion .
Ejecutar el contenedor:docker run -p 5000:5000 \
  -e MODEL_TO_LOAD="istrats_pretrained" \
  --name predictor-web \
  mi-app-prediccion
Ahora la aplicación estará corriendo dentro de un contenedor Docker, pero seguirá siendo accesible en http://localhost:5000.
