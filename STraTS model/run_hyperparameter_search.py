#!/usr/bin/env python3
"""
Script para realizar una búsqueda de hiperparámetros (Grid Search) para los modelos STraTS.

Este script lee un archivo de configuración con espacios de búsqueda, genera todas las
combinaciones de hiperparámetros y ejecuta un experimento de entrenamiento para cada una,
guardando al final un resumen en formato CSV.
"""
import os
import sys
import json
import argparse
import logging
import itertools
import pandas as pd
from typing import Dict, Any, List

# Importar la función principal de entrenamiento desde el script existente
# Es crucial que run_experiment devuelva los resultados de la evaluación.
from train import run_experiment

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("hyperparam_search.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def generate_hyperparameter_combinations(search_space: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Genera todas las combinaciones posibles de hiperparámetros a partir del espacio de búsqueda.
    """
    model_params = search_space.get('model', {})
    training_params = search_space.get('training', {})

    keys = list(model_params.keys()) + list(training_params.keys())
    value_lists = list(model_params.values()) + list(training_params.values())

    # Generar el producto cartesiano de los valores
    combinations = list(itertools.product(*value_lists))

    param_combinations = []
    for combo in combinations:
        param_dict = dict(zip(keys, combo))
        param_combinations.append(param_dict)

    logger.info(f"Generadas {len(param_combinations)} combinaciones de hiperparámetros.")
    return param_combinations

def merge_configs(run_params: Dict[str, Any], fixed_params: Dict[str, Any], search_space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combina los parámetros de la ejecución actual con los parámetros fijos,
    usando el search_space para determinar la estructura correcta.
    """
    config = json.loads(json.dumps(fixed_params)) # Deep copy

    # Asegurarse de que las claves 'model' y 'training' existen en el diccionario final
    if 'model' not in config:
        config['model'] = {}
    if 'training' not in config:
        config['training'] = {}

    # Obtener las claves que pertenecen a 'model' y 'training' desde el search_space
    model_search_keys = search_space.get('model', {}).keys()
    training_search_keys = search_space.get('training', {}).keys()

    # Asignar cada parámetro de la ejecución actual a la sección correcta
    for key, value in run_params.items():
        if key in model_search_keys:
            config['model'][key] = value
        elif key in training_search_keys:
            config['training'][key] = value
            
    return config

def run_search(model_type: str, config_path: str, base_output_dir: str, data_path: str, pretrained_path: str = None):
    """
    Función principal para ejecutar la búsqueda de hiperparámetros.
    """
    logger.info(f"--- Iniciando Búsqueda de Hiperparámetros para el modelo: {model_type.upper()} ---")
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: No se encontró el archivo de configuración en '{config_path}'")
        return
    
    search_space = config_data.get('hyperparameter_search_space', {})
    fixed_params = config_data.get('fixed_parameters', {})

    if not search_space:
        logger.error("La clave 'hyperparameter_search_space' no se encontró en el archivo de configuración.")
        return

    param_combinations = generate_hyperparameter_combinations(search_space)

    os.makedirs(base_output_dir, exist_ok=True)
    
    results_summary = []

    for i, run_params in enumerate(param_combinations):
        run_name = f"run_{i+1:03d}"
        output_dir = os.path.join(base_output_dir, run_name)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Iniciando {run_name} / {len(param_combinations)} con parámetros:")
        logger.info(json.dumps(run_params, indent=2))
        logger.info(f"{'='*80}")
        
        # CORRECCIÓN: Pasar el search_space a merge_configs
        experiment_config = merge_configs(run_params, fixed_params, search_space)
        
        try:
            evaluation_results = run_experiment(
                model_type=model_type,
                config=experiment_config,
                output_dir=output_dir,
                data_path=data_path,
                pretrained_path=pretrained_path
            )

            if evaluation_results:
                summary_item = run_params.copy()
                summary_item['run_name'] = run_name
                
                val_metrics = evaluation_results.get('val', {})
                test_metrics = evaluation_results.get('test', {})
                
                summary_item['val_auc'] = val_metrics.get('AUC', 0.0)
                summary_item['val_auprc'] = val_metrics.get('AUPRC', 0.0)
                summary_item['val_f1'] = val_metrics.get('F1-score', 0.0)
                summary_item['test_auc'] = test_metrics.get('AUC', 0.0)
                summary_item['test_auprc'] = test_metrics.get('AUPRC', 0.0)
                summary_item['test_f1'] = test_metrics.get('F1-score', 0.0)
                
                results_summary.append(summary_item)
            else:
                logger.warning(f"La ejecución {run_name} no devolvió resultados.")

        except Exception as e:
            logger.error(f"Error crítico en {run_name}: {e}", exc_info=True)
            logger.error("Saltando a la siguiente combinación.")

    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_df = summary_df.sort_values(by='val_auprc', ascending=False)
        
        summary_path = os.path.join(base_output_dir, 'hyperparameter_search_summary.csv')
        try:
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"\n--- Búsqueda de Hiperparámetros Finalizada ---")
            logger.info(f"Resumen de resultados guardado exitosamente en: {summary_path}")
            logger.info("\nMejores 5 combinaciones (ordenadas por Val AUPRC):")
            logger.info(f"\n{summary_df.head(5).to_string()}")
        except Exception as e:
            logger.error(f"No se pudo guardar el resumen en CSV: {e}")
    else:
        logger.warning("No se completó ninguna ejecución exitosamente. No se generó resumen en CSV.")

def main():
    parser = argparse.ArgumentParser(description="Búsqueda de Hiperparámetros para STraTS")
    parser.add_argument('--model_type', type=str, required=True, choices=['strats', 'istrats'])
    parser.add_argument('--config', type=str, default='strats_config_hyperopt.json')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='strats_data')
    parser.add_argument('--pretrained_path', type=str, default=None)
    args = parser.parse_args()

    run_search(
        model_type=args.model_type,
        config_path=args.config,
        base_output_dir=args.output_dir,
        data_path=args.data_path,
        pretrained_path=args.pretrained_path
    )

if __name__ == '__main__':
    main()

