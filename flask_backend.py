#!/usr/bin/env python3
"""
Backend Flask para la aplicación web de predicción de pacientes.
Permite cargar dinámicamente diferentes versiones del modelo (strats, istrats, etc.)
mediante una variable de entorno.

Uso:
  # Cargar el modelo por defecto (strats_pretrained)
  python flask_backend.py

  # Cargar un modelo específico (p.ej. istrats_scratch)
  MODEL_TO_LOAD=istrats_scratch python flask_backend.py
"""
import os
import logging
import json
import pickle
import traceback

import torch
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- Importar Clases de Modelo desde el módulo centralizado ---
from models import STraTSModel, iSTraTSModel

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='.')
CORS(app)

# --- Variables Globales ---
model = None
metadata = None
device = None

def load_model_and_metadata(model_name: str, device: str):
    """
    Carga dinámicamente un modelo entrenado y sus metadatos.
    El nombre del modelo determina la carpeta de resultados y la clase a usar.
    
    Args:
        model_name (str): Identificador del modelo (p.ej., 'strats_pretrained').
        device (str): Dispositivo ('cpu' o 'cuda').
        
    Returns:
        Tuple[nn.Module, dict]: El modelo cargado y los metadatos.
    """
    results_dir = os.path.join('results', model_name)
    logger.info(f"Intentando cargar artefactos del modelo '{model_name}' desde: {results_dir}")

    # --- Cargar Metadatos ---
    metadata_path = os.path.join('strats_data', 'metadata.pkl')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Archivo de metadatos no encontrado en: {metadata_path}")
    
    with open(metadata_path, 'rb') as f:
        loaded_metadata = pickle.load(f)
    logger.info("✓ Metadatos cargados correctamente.")

    # --- Cargar Modelo ---
    model_path = os.path.join(results_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Archivo de modelo no encontrado en: {model_path}")

    # Determinar la clase del modelo a partir del nombre
    if 'istrats' in model_name:
        model_class = iSTraTSModel
    elif 'strats' in model_name:
        model_class = STraTSModel
    else:
        raise ValueError(f"No se pudo determinar la clase del modelo para '{model_name}'. El nombre debe contener 'strats' o 'istrats'.")
    
    logger.info(f"Usando la clase de modelo: {model_class.__name__}")

    # Cargar configuración del modelo
    config_path = 'strats_config.json'
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)['model']
    except FileNotFoundError:
        logger.warning(f"ADVERTENCIA: No se encontró {config_path}. Usando hiperparámetros por defecto.")
        config = {'d_model': 128, 'n_heads': 16, 'n_layers': 8, 'dropout': 0.15}

    # Instanciar modelo
    loaded_model = model_class(
        feature_dim=loaded_metadata['n_features'],
        max_seq_length=loaded_metadata['max_seq_length'],
        **config
    )
    
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    loaded_model.to(device)
    loaded_model.eval()
    
    logger.info(f"✓ Modelo '{model_name}' cargado y listo para predicción en dispositivo '{device}'.")
    
    return loaded_model, loaded_metadata

def preprocess_new_patient(patient_data, metadata):
    """
    Convierte los datos brutos de un paciente al formato de tensor que el modelo necesita.
    """
    feature_to_idx = metadata['feature_to_idx']
    scalers = metadata.get('scalers', {})
    max_seq_length = metadata['max_seq_length']
    n_features = metadata['n_features']
    
    X = np.zeros((1, max_seq_length, n_features), dtype=np.float32)
    times = np.zeros((1, max_seq_length), dtype=np.float32)
    masks = np.zeros((1, max_seq_length, n_features), dtype=np.float32)
    
    data_by_time = {}
    for time, param, value in patient_data:
        if value is None or value == '': continue
        if time not in data_by_time: data_by_time[time] = []
        data_by_time[time].append((param, float(value)))
        
    for t_idx, time_val in enumerate(sorted(data_by_time.keys())):
        if t_idx >= max_seq_length: break
        times[0, t_idx] = time_val
        for param, value in data_by_time[time_val]:
            if param in feature_to_idx:
                f_idx = feature_to_idx[param]
                value_scaled = scalers[param].transform(np.array([[value]]))[0, 0] if param in scalers else value
                X[0, t_idx, f_idx] = value_scaled
                masks[0, t_idx, f_idx] = 1.0
    
    return torch.FloatTensor(X), torch.FloatTensor(times), torch.FloatTensor(masks)

@app.route('/predict', methods=['POST'])
def handle_predict():
    """
    Endpoint para recibir datos de un paciente y devolver una predicción.
    """
    try:
        if model is None or metadata is None:
            return jsonify({'error': 'El modelo no está disponible en el servidor.'}), 503
        
        data = request.get_json()
        if not data or 'patient_data' not in data:
            return jsonify({'error': 'Formato de datos inválido.'}), 400
        
        X_tensor, times_tensor, mask_tensor = preprocess_new_patient(data['patient_data'], metadata)
        
        with torch.no_grad():
            logits = model(X_tensor.to(device), times_tensor.to(device), mask_tensor.to(device))
            probabilities = torch.softmax(logits, dim=1)
        
        prob_evento = probabilities[0, 1].item()
        predicted_class = probabilities.argmax(dim=1).item()
        
        return jsonify({
            'prediction': predicted_class,
            'probability': prob_evento,
            'prediction_label': 'Evento' if predicted_class == 1 else 'No Evento',
            'probability_percent': f"{prob_evento*100:.2f}%"
        })
        
    except Exception as e:
        logger.error(f"Error en /predict: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500

@app.route('/')
def index():
    return render_template('patient_prediction_webapp.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'metadata_loaded': metadata is not None,
        'device': str(device),
        'loaded_model_name': os.environ.get('MODEL_TO_LOAD', 'strats_pretrained')
    })

def initialize_app():
    """Inicializa el modelo y las variables globales al arrancar el servidor."""
    global model, metadata, device
    
    # --- Carga dinámica del modelo ---
    # Lee la variable de entorno 'MODEL_TO_LOAD'. Si no existe, usa 'strats_pretrained' por defecto.
    model_to_load = os.environ.get('MODEL_TO_LOAD', 'strats_pretrained')
    
    # Valida que el nombre sea uno de los esperados
    valid_models = ['strats_scratch', 'istrats_scratch', 'strats_pretrained', 'istrats_pretrained']
    if model_to_load not in valid_models:
        logger.error(f"FATAL: El valor de MODEL_TO_LOAD ('{model_to_load}') no es válido.")
        logger.error(f"Valores permitidos son: {', '.join(valid_models)}")
        # En un entorno de producción, es mejor que la aplicación no arranque si la config es inválida.
        raise ValueError(f"Variable de entorno MODEL_TO_LOAD no válida: {model_to_load}")

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Inicializando aplicación en dispositivo: {device}")
        
        model, metadata = load_model_and_metadata(model_to_load, device)
        
    except Exception as e:
        logger.error(f"FATAL: No se pudo inicializar el modelo '{model_to_load}' en el arranque: {e}")
        logger.error("El servidor arrancará, pero las predicciones fallarán. Revisa la configuración y las rutas.")

if __name__ == '__main__':
    initialize_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
