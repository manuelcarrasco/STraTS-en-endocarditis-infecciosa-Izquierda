#!/usr/bin/env python3
"""
Módulo para calcular los "Contribution Scores" para el modelo iSTraTS.
Versión corregida y ampliada que:
1. Desglosa la contribución de variables categóricas por cada categoría.
2. Mantiene la generación de gráficos de contribución temporal por paciente.
3. AÑADIDO: Genera una tabla DataFrame con la media y desviación estándar
   de las contribuciones por variable para todos los individuos.
4. CORRECCIÓN: Se modifica el cálculo de la contribución para usar directamente
   la magnitud del embedding del triplete, evitando el uso de los pesos de
   atención que causaban contribuciones idénticas.
"""
import os
import json
import logging
from typing import Dict, List, Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def create_contribution_table(all_contributions_dict: Dict[str, List[float]], output_dir: str) -> pd.DataFrame:
    """
    Crea, guarda y devuelve una tabla de pandas con la media y la desviación
    estándar de los scores de contribución para cada variable.

    Args:
        all_contributions_dict (Dict[str, List[float]]): Un diccionario donde las claves son
                                                          los nombres de las variables y los
                                                          valores son listas de todas las
                                                          contribuciones observadas.
        output_dir (str): Directorio donde guardar la tabla en formato CSV.

    Returns:
        pd.DataFrame: Un DataFrame de pandas con las estadísticas, ordenado por
                      la contribución media de forma descendente.
    """
    logger.info("Creando tabla de resumen de contribuciones...")

    # Crear una lista de diccionarios para construir el DataFrame
    stats_list = []
    for feature_name, contributions in all_contributions_dict.items():
        if contributions:
            stats_list.append({
                'Variable': feature_name,
                'Contribucion_Media': np.mean(contributions),
                'Contribucion_Std': np.std(contributions),
                'Observaciones': len(contributions)
            })

    if not stats_list:
        logger.warning("No se encontraron contribuciones para generar la tabla.")
        return pd.DataFrame()

    # Crear el DataFrame y ordenarlo
    df_stats = pd.DataFrame(stats_list)
    df_stats = df_stats.sort_values(by='Contribucion_Media', ascending=False).reset_index(drop=True)

    # Guardar la tabla en un archivo CSV
    table_path = os.path.join(output_dir, 'istrats_contribution_summary_table.csv')
    try:
        df_stats.to_csv(table_path, index=False, sep=';', decimal=',')
        logger.info(f"Tabla de contribuciones guardada exitosamente en: {table_path}")
    except Exception as e:
        logger.error(f"No se pudo guardar la tabla de contribuciones: {e}")

    # Mostrar las 15 variables más importantes en el log
    logger.info("--- Top 15 Variables por Contribución Media ---")
    logger.info("\n" + df_stats.head(15).to_string())
    
    return df_stats


def calculate_global_contribution_scores(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    feature_names: List[str],
    device: str,
    output_dir: str,
    scalers: Dict[str, StandardScaler] = None,
    normalize_scores: bool = True
) -> Tuple[Dict[str, Dict], pd.DataFrame]:
    """
    Calcula los scores de contribución globales y la tabla de estadísticas.
    """
    logger.info("=== Iniciando cálculo de Contribution Scores (desglosado por categoría) ===")
    model.eval()
    model.to(device)

    n_features = len(feature_names)
    
    # Diccionario para acumular todas las contribuciones por variable para la tabla
    all_contributions_for_table = {name: [] for name in feature_names}
    
    # Listas para acumular valores para el gráfico
    all_contributions_for_plot = [[] for _ in range(n_features)]
    all_raw_values_for_plot = [[] for _ in range(n_features)]

    with torch.no_grad():
        for batch in data_loader:
            X, times, mask = batch['X'].to(device), batch['times'].to(device), batch['mask'].to(device)
            
            # --- LÓGICA DE CONTRIBUCIÓN CORREGIDA ---
            # 1. Obtener los embeddings de cada componente del triplete
            var_indices = torch.arange(model.feature_dim).to(device)
            variable_emb = model.variable_emb(var_indices).view(1, 1, model.feature_dim, model.d_model)
            time_emb = model.cve_time(times.unsqueeze(-1)).unsqueeze(2)
            value_emb = model.cve_value(X.unsqueeze(-1))
            
            # 2. Sumar los embeddings para formar el triplete completo
            full_triplet_emb = variable_emb + time_emb + value_emb
            
            # 3. Calcular la norma L2 como score de contribución
            #    Esta es la corrección clave: la contribución es la magnitud del embedding
            #    de cada observación individual, sin usar los pesos de atención.
            contribution_this_batch = torch.norm(full_triplet_emb, p=2, dim=-1)

            # Acumular contribuciones y valores donde hay datos (usando la máscara)
            for i in range(n_features):
                feature_mask = mask[:, :, i].bool()
                if feature_mask.any():
                    # Aplicar la máscara a las contribuciones calculadas
                    contributions = contribution_this_batch[:, :, i][feature_mask].cpu().numpy()
                    raw_values = X[:, :, i][feature_mask].cpu().numpy()

                    # Para el gráfico
                    all_contributions_for_plot[i].extend(contributions)
                    all_raw_values_for_plot[i].extend(raw_values)
                    
                    # Para la tabla (agregando por nombre de variable)
                    feature_name = feature_names[i]
                    all_contributions_for_table[feature_name].extend(contributions)

    # --- NUEVO: Crear la tabla de contribuciones ---
    contribution_table = create_contribution_table(all_contributions_for_table, output_dir)

    # --- Procesar resultados para el gráfico (como antes) ---
    final_results_plot = {}
    for i, name in enumerate(feature_names):
        if not all_contributions_for_plot[i]:
            continue

        contributions = np.array(all_contributions_for_plot[i])
        raw_values_normalized = np.array(all_raw_values_for_plot[i])
        
        is_continuous = scalers and name in scalers

        if is_continuous:
            score = np.mean(contributions)
            value_range = "N/A"
            if len(contributions) > 1:
                top_percentile_threshold = np.percentile(contributions, 75)
                top_values_normalized = raw_values_normalized[contributions >= top_percentile_threshold]
                
                scaler = scalers[name]
                top_values_unnormalized = scaler.inverse_transform(top_values_normalized.reshape(-1, 1)).flatten()

                if len(top_values_unnormalized) > 1:
                    p25, p75 = np.percentile(top_values_unnormalized, [25, 75])
                    value_range = f"[{p25:.2f} - {p75:.2f}]"
                elif len(top_values_unnormalized) == 1:
                    value_range = f"{top_values_unnormalized[0]:.2f}"
            final_results_plot[name] = {"score": float(score), "value_range": value_range, "is_categorical": False}
        else:
            unique_categories = np.unique(raw_values_normalized)
            for category in unique_categories:
                category_mask = raw_values_normalized == category
                if np.any(category_mask):
                    avg_score = np.mean(contributions[category_mask])
                    feature_category_name = f"{name} (cat. {int(category)})"
                    final_results_plot[feature_category_name] = {"score": float(avg_score), "is_categorical": True}

    all_scores = [res["score"] for res in final_results_plot.values() if "score" in res and res["score"] > 0]
    if normalize_scores and all_scores:
        max_score = max(all_scores)
        if max_score > 0:
            for key in final_results_plot:
                final_results_plot[key]["score"] /= max_score

    sorted_results = dict(sorted(final_results_plot.items(), key=lambda item: item[1]["score"], reverse=True))
    
    logger.info("--- Top 10 Contribution Scores (para gráfico) ---")
    for i, (feature, data) in enumerate(list(sorted_results.items())[:10]):
        logger.info(f"{i+1}. {feature}: Score={data['score']:.4f}")

    filename_suffix = "normalized" if normalize_scores else "raw"
    results_path = os.path.join(output_dir, f'istrats_global_contribution_{filename_suffix}.json')
    with open(results_path, 'w') as f:
        json.dump(sorted_results, f, indent=4)
    logger.info(f"Resultados para gráfico guardados en: {results_path}")

    plot_global_contribution_scores(sorted_results, output_dir, filename_suffix, normalize_scores)
    
    return sorted_results, contribution_table


def plot_global_contribution_scores(scores: Dict[str, Dict], output_dir: str, filename_suffix: str, normalized: bool, top_k: int = 20):
    """Visualiza los scores de contribución globales de iSTraTS."""
    logger.info(f"Generando gráfico de Contribution Scores Globales (Top {top_k})...")
    
    filtered_scores = {k: v for k, v in scores.items() if v['score'] > 1e-6}
    top_scores = dict(list(filtered_scores.items())[:top_k])
    
    if not top_scores:
        logger.warning("No hay características con score de contribución > 0 para graficar.")
        return

    feature_labels = []
    for name, data in top_scores.items():
        if data.get("is_categorical", False):
            feature_labels.append(name)
        else:
            feature_labels.append(f"{name}\n{data.get('value_range', 'N/A')}")

    score_values = [data['score'] for data in top_scores.values()]
    
    plt.figure(figsize=(12, 10))
    df = pd.DataFrame({'Feature': feature_labels, 'Contribution Score': score_values})
    sns.barplot(x='Contribution Score', y='Feature', data=df, palette='viridis')
    
    title = f'Top {len(top_scores)} iSTraTS Global Feature Contribution (by Category)'
    xlabel = 'Average Contribution Score'
    if normalized:
        title += " (Normalized)"
        xlabel = 'Normalized ' + xlabel

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Feature / Feature: Category', fontsize=12)
    
    filepath = os.path.join(output_dir, f'istrats_global_contribution_{filename_suffix}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico de scores de contribución global guardado en: {filepath}")

def analyze_and_plot_patient_contributions(
    model: torch.nn.Module,
    patient_data: Dict,
    patient_id: str,
    feature_names: List[str],
    device: str,
    output_dir: str,
    scalers: Dict[str, StandardScaler] = None,
    top_k: int = 8
):
    """
    Analiza y visualiza las contribuciones temporales para un único paciente.
    """
    model.eval()
    model.to(device)

    X = patient_data['X'].unsqueeze(0).to(device)
    times = patient_data['times'].unsqueeze(0).to(device)
    mask = patient_data['mask'].unsqueeze(0).to(device)

    with torch.no_grad():
        # Lógica de contribución corregida también para el análisis por paciente
        var_indices = torch.arange(model.feature_dim).to(device)
        variable_emb = model.variable_emb(var_indices).view(1, 1, model.feature_dim, model.d_model)
        time_emb = model.cve_time(times.unsqueeze(-1)).unsqueeze(2)
        value_emb = model.cve_value(X.unsqueeze(-1))
        
        full_triplet_emb = variable_emb + time_emb + value_emb
        # La contribución es la magnitud del embedding del triplete
        contributions = torch.norm(full_triplet_emb, p=2, dim=-1).squeeze(0).cpu().numpy()


    X_unnormalized = X.squeeze(0).cpu().numpy().copy()
    if scalers:
        for i, name in enumerate(feature_names):
            if name in scalers:
                scaler = scalers[name]
                feature_mask = mask.squeeze(0)[:, i].bool().cpu().numpy()
                if np.any(feature_mask):
                    vals_norm = X_unnormalized[feature_mask, i]
                    X_unnormalized[feature_mask, i] = scaler.inverse_transform(vals_norm.reshape(-1, 1)).flatten()

    # Aplicar la máscara a las contribuciones para el ranking
    masked_contributions = contributions * mask.squeeze(0).cpu().numpy()
    max_contributions_per_feature = np.max(masked_contributions, axis=0)
    top_feature_indices = np.argsort(max_contributions_per_feature)[-top_k:][::-1]

    patient_plot_dir = os.path.join(output_dir, 'patient_temporal_plots', str(patient_id))
    os.makedirs(patient_plot_dir, exist_ok=True)
    
    logger.info(f"Generando {top_k} gráficos temporales para el paciente: {patient_id}")
    for feat_idx in top_feature_indices:
        feat_name = feature_names[feat_idx]
        feat_mask = mask.squeeze(0)[:, feat_idx].bool().cpu().numpy()

        if not np.any(feat_mask):
            continue

        time_steps = times.squeeze(0)[feat_mask].cpu().numpy()
        feat_values = X_unnormalized[feat_mask, feat_idx]
        feat_contributions = contributions[feat_mask, feat_idx]

        fig, ax1 = plt.subplots(figsize=(8, 5))

        color = 'tab:blue'
        ax1.set_xlabel('Time')
        ax1.set_ylabel(f'Measured Value ({feat_name})', color=color)
        ax1.plot(time_steps, feat_values, color=color, marker='o', label='Measured Value')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Contribution Score', color=color)
        ax2.plot(time_steps, feat_contributions, color=color, marker='x', linestyle='--', label='Contribution Score')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f'Temporal Contribution for Patient {patient_id}\nFeature: {feat_name}')
        fig.tight_layout()
        
        plot_path = os.path.join(patient_plot_dir, f'{feat_name.replace("/", "_")}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

def run_istrats_analysis(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    feature_names: List[str],
    device: str,
    output_dir: str,
    scalers: Dict[str, StandardScaler] = None,
    num_patients_to_plot: int = 8
):
    """
    Función orquestadora que ejecuta TODOS los análisis:
    1. El score de contribución global y la tabla de resumen.
    2. Los gráficos de contribución temporal para pacientes individuales.
    """
    logger.info("====== INICIANDO ANÁLISIS DE INTERPRETABILIDAD DE iSTraTS ======")
    
    # --- PASO 1: Calcular scores globales, gráficos y la tabla de resumen ---
    _, _ = calculate_global_contribution_scores(
        model=model, data_loader=test_loader, feature_names=feature_names,
        device=device, output_dir=output_dir, scalers=scalers
    )

    # --- PASO 2: Analizar y graficar contribuciones temporales para pacientes individuales ---
    logger.info(f"\n--- Analizando contribuciones temporales para {num_patients_to_plot} pacientes de ejemplo ---")
    if not hasattr(test_loader.dataset, 'patient_ids'):
        logger.warning("El dataset de test no tiene 'patient_ids'. No se pueden generar plots por paciente.")
        return

    all_patient_ids = test_loader.dataset.patient_ids
    # Asegurarse de no intentar acceder a más pacientes de los que hay
    num_to_plot = min(num_patients_to_plot, len(all_patient_ids))
    
    # Seleccionar pacientes aleatoriamente para más variabilidad
    if len(all_patient_ids) > num_to_plot:
        patient_indices_to_plot = np.random.choice(len(all_patient_ids), num_to_plot, replace=False)
    else:
        patient_indices_to_plot = np.arange(len(all_patient_ids))


    for i in patient_indices_to_plot:
        patient_sample = test_loader.dataset[i]
        patient_id = all_patient_ids[i]
        
        analyze_and_plot_patient_contributions(
            model=model,
            patient_data=patient_sample,
            patient_id=patient_id,
            feature_names=feature_names,
            device=device,
            output_dir=output_dir,
            scalers=scalers,
            top_k=8
        )
    logger.info("====== ANÁLISIS DE INTERPRETABILIDAD DE iSTraTS FINALIZADO ======")




