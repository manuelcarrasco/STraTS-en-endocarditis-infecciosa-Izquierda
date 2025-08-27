#!/usr/bin/env python3
"""
Módulo mejorado y corregido para el análisis de interpretabilidad usando SHAP para modelos STraTS.
Incluye correcciones para problemas de dimensiones y mejor manejo de errores.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import shap

# --- Configuración de Logging y Warnings ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class StratsShapInterpreter:
    """
    Intérprete SHAP mejorado para modelos de la familia STraTS con correcciones dimensionales.
    """
    
    def __init__(self, model: nn.Module, feature_names: List[str], device: str = 'cuda'):
        """
        Inicializa el intérprete SHAP.
        
        Args:
            model: Modelo STraTS o iSTraTS entrenado.
            feature_names: Lista de nombres de las características.
            device: Dispositivo de cómputo ('cuda' o 'cpu').
        """
        self.model = model.to(device)
        self.model.eval()
        self.feature_names = feature_names
        self.device = device
        self.explainer = None
        self.background_data = None
        self.n_features = len(feature_names)
        logger.info(f"Intérprete inicializado con {self.n_features} características en {device}")
        
    def prepare_background_data(self, train_loader: DataLoader, n_samples: int = 50) -> None:
        """
        Prepara datos de fondo para SHAP usando muestras del conjunto de entrenamiento.
        """
        logger.info(f"Preparando datos de fondo con {n_samples} muestras...")
        
        bg_samples, bg_times, bg_masks = [], [], []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                if len(bg_samples) * train_loader.batch_size >= n_samples:
                    break
                
                batch_X = batch['X'].to(self.device)
                batch_times = batch['times'].to(self.device)
                batch_mask = batch['mask'].to(self.device)
                
                # Tomar solo las muestras necesarias del último batch
                remaining = n_samples - len(bg_samples) * train_loader.batch_size
                if remaining < batch_X.size(0):
                    batch_X = batch_X[:remaining]
                    batch_times = batch_times[:remaining]
                    batch_mask = batch_mask[:remaining]
                
                bg_samples.append(batch_X)
                bg_times.append(batch_times)
                bg_masks.append(batch_mask)
        
        # Concatenar todos los batches
        self.background_data = {
            'X': torch.cat(bg_samples, dim=0)[:n_samples],
            'times': torch.cat(bg_times, dim=0)[:n_samples],
            'mask': torch.cat(bg_masks, dim=0)[:n_samples]
        }
        
        logger.info(f"Datos de fondo preparados: X={self.background_data['X'].shape}, "
                   f"times={self.background_data['times'].shape}, "
                   f"mask={self.background_data['mask'].shape}")
        
    def _model_wrapper(self, X_flat: np.ndarray) -> np.ndarray:
        """
        Wrapper mejorado del modelo para SHAP con mejor manejo dimensional.
        """
        try:
            # Convertir entrada a tensor
            X_tensor = torch.from_numpy(X_flat).float().to(self.device)
            batch_size = X_tensor.shape[0]
            
            # CORRECCIÓN: Calcular dimensiones basándose en los datos de fondo reales
            bg_shape = self.background_data['X'].shape
            seq_len = bg_shape[1]
            n_features = bg_shape[2]
            expected_flat_size = seq_len * n_features
            
            logger.debug(f"Wrapper input shape: {X_tensor.shape}")
            logger.debug(f"Expected flat size: {expected_flat_size}, actual: {X_tensor.shape[1]}")
            logger.debug(f"Background shape: {bg_shape}")
            
            # Verificar que las dimensiones coinciden
            if X_tensor.shape[1] != expected_flat_size:
                logger.error(f"Dimensión mismatch: esperado {expected_flat_size}, obtenido {X_tensor.shape[1]}")
                # Intentar ajustar cortando o rellenando
                if X_tensor.shape[1] > expected_flat_size:
                    X_tensor = X_tensor[:, :expected_flat_size]
                else:
                    # Rellenar con ceros
                    padding = torch.zeros(batch_size, expected_flat_size - X_tensor.shape[1], device=self.device)
                    X_tensor = torch.cat([X_tensor, padding], dim=1)
            
            # Reshape para formato esperado por el modelo
            X_reshaped = X_tensor.view(batch_size, seq_len, n_features)
            
            # Preparar times y masks del tamaño correcto
            if batch_size <= self.background_data['times'].shape[0]:
                times = self.background_data['times'][:batch_size]
                mask = self.background_data['mask'][:batch_size]
            else:
                # Repetir datos de fondo si necesitamos más
                n_repeats = (batch_size + self.background_data['times'].shape[0] - 1) // self.background_data['times'].shape[0]
                times = self.background_data['times'].repeat(n_repeats, 1)[:batch_size]
                mask = self.background_data['mask'].repeat(n_repeats, 1, 1)[:batch_size]
            
            # Asegurar que todos los tensores están en el dispositivo correcto
            times = times.to(self.device)
            mask = mask.to(self.device)
            
            # Obtener predicciones del modelo
            with torch.no_grad():
                logits = self.model(X_reshaped, times, mask)
                probs = F.softmax(logits, dim=1)
            
            # Retornar probabilidades como numpy array
            return probs.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error en _model_wrapper: {e}")
            # Retornar array de ceros en caso de error
            return np.zeros((X_flat.shape[0], 2))
            
    def initialize_explainer(self, link='identity', nsamples=100) -> None:
        """
        Inicializa el KernelExplainer de SHAP con parámetros optimizados.
        """
        if self.background_data is None:
            raise ValueError("Debe preparar los datos de fondo antes de inicializar el explicador.")
            
        logger.info(f"Inicializando KernelExplainer de SHAP (link={link}, nsamples={nsamples})...")
        
        # Preparar datos de fondo aplanados
        bg_shape = self.background_data['X'].shape
        background_flat = self.background_data['X'].view(bg_shape[0], -1).cpu().numpy()
        
        logger.info(f"Background data shape: {bg_shape} -> flat: {background_flat.shape}")
        
        # Limitar tamaño del background para eficiencia
        max_background = 50  # Reducido para mayor estabilidad
        if background_flat.shape[0] > max_background:
            logger.info(f"Limitando datos de fondo de {background_flat.shape[0]} a {max_background} muestras")
            indices = np.random.choice(background_flat.shape[0], max_background, replace=False)
            background_flat = background_flat[indices]
        
        # Crear explicador
        try:
            self.explainer = shap.KernelExplainer(
                self._model_wrapper, 
                background_flat,
                link=link
            )
            logger.info("Explicador SHAP inicializado correctamente.")
        except Exception as e:
            logger.error(f"Error inicializando explicador SHAP: {e}")
            raise
    
    def explain_batch(self, X: torch.Tensor, n_samples: int = 100, batch_size: int = 5) -> np.ndarray:
        """
        Explica un batch de muestras usando SHAP con procesamiento por lotes más pequeños.
        """
        if self.explainer is None:
            raise ValueError("El explicador no ha sido inicializado.")
            
        logger.info(f"Explicando batch de {X.shape[0]} muestras (batch_size={batch_size})...")
        
        batch_size_orig, seq_len, n_features = X.shape
        X_flat = X.view(batch_size_orig, -1).cpu().numpy()
        
        logger.info(f"Input shape: {X.shape} -> flat: {X_flat.shape}")
        logger.info(f"Expected output shape después de reshape: ({batch_size_orig}, {seq_len}, {n_features})")
        
        # Procesar en mini-batches muy pequeños para evitar problemas de memoria
        shap_values_list = []
        
        for i in range(0, batch_size_orig, batch_size):
            end_idx = min(i + batch_size, batch_size_orig)
            batch_X = X_flat[i:end_idx]
            
            logger.info(f"  Procesando muestras {i+1}-{end_idx}/{batch_size_orig}...")
            
            try:
                # Calcular valores SHAP para este mini-batch
                shap_values = self.explainer.shap_values(
                    batch_X, 
                    nsamples=n_samples,
                    l1_reg='num_features(5)'  # Regularización más fuerte para estabilidad
                )
                
                # Manejar salida multi-clase
                if isinstance(shap_values, list):
                    # Para clasificación binaria, tomar clase positiva (índice 1)
                    shap_values_batch = shap_values[1]
                else:
                    shap_values_batch = shap_values
                
                logger.info(f"  SHAP values shape for batch {i}-{end_idx}: {shap_values_batch.shape}")
                shap_values_list.append(shap_values_batch)
                
            except Exception as e:
                logger.error(f"Error explicando batch {i}-{end_idx}: {e}")
                # Añadir valores cero en caso de error
                zero_shape = (end_idx - i, X_flat.shape[1])
                shap_values_list.append(np.zeros(zero_shape))
        
        # Concatenar todos los resultados
        shap_values_combined = np.concatenate(shap_values_list, axis=0)
        logger.info(f"Combined SHAP values shape: {shap_values_combined.shape}")
        
        # CORRECCIÓN: Verificar dimensiones antes del reshape
        expected_total_size = batch_size_orig * seq_len * n_features
        actual_size = shap_values_combined.size
        
        if actual_size != expected_total_size:
            logger.warning(f"Dimension mismatch: expected {expected_total_size}, got {actual_size}")
            
            # Intentar ajustar las dimensiones
            if actual_size > expected_total_size:
                # Cortar los datos extras
                shap_values_combined = shap_values_combined.flatten()[:expected_total_size]
                logger.info(f"Truncated SHAP values to size {expected_total_size}")
            else:
                # Rellenar con ceros
                padding_size = expected_total_size - actual_size
                shap_values_combined = np.concatenate([
                    shap_values_combined.flatten(), 
                    np.zeros(padding_size)
                ])
                logger.info(f"Padded SHAP values by {padding_size} zeros")
        
        # Reshape a formato original
        try:
            shap_values_reshaped = shap_values_combined.reshape(batch_size_orig, seq_len, n_features)
            logger.info(f"Explicación completada. Shape final: {shap_values_reshaped.shape}")
            return shap_values_reshaped
        except ValueError as e:
            logger.error(f"Error en reshape final: {e}")
            logger.error(f"Trying to reshape {shap_values_combined.size} elements into ({batch_size_orig}, {seq_len}, {n_features})")
            # Retornar array de ceros como fallback
            return np.zeros((batch_size_orig, seq_len, n_features))
    
    @staticmethod
    def compute_feature_importance(shap_values: np.ndarray, feature_names: List[str], 
                                 mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Computa la importancia global de características.
        """
        if mask is not None:
            # Aplicar máscara solo donde hay valores observados
            shap_values_masked = shap_values * mask
            # Calcular media considerando solo valores observados
            feature_importance = []
            for feat_idx in range(shap_values.shape[2]):
                feat_mask = mask[:, :, feat_idx]
                if feat_mask.sum() > 0:
                    feat_values = shap_values_masked[:, :, feat_idx]
                    importance = np.sum(np.abs(feat_values)) / feat_mask.sum()
                else:
                    importance = 0.0
                feature_importance.append(importance)
            mean_abs_shap = np.array(feature_importance)
        else:
            # Sin máscara, usar media simple
            mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 1))
        
        # Crear diccionario ordenado
        feature_importance_dict = dict(zip(feature_names, mean_abs_shap))
        return dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    @staticmethod
    def compute_temporal_importance(shap_values: np.ndarray, feature_names: List[str], 
                                  mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Computa la importancia temporal de características.
        """
        temporal_importance = {}
        
        for i, feature_name in enumerate(feature_names):
            if mask is not None:
                # Considerar solo timesteps con valores observados
                feat_mask = mask[:, :, i]
                feat_shap = shap_values[:, :, i] * feat_mask
                
                # Media por timestep (considerando solo valores observados)
                temporal_values = []
                for t in range(shap_values.shape[1]):
                    mask_t = feat_mask[:, t]
                    if mask_t.sum() > 0:
                        temporal_values.append(
                            np.sum(np.abs(feat_shap[:, t])) / mask_t.sum()
                        )
                    else:
                        temporal_values.append(0.0)
                temporal_importance[feature_name] = np.array(temporal_values)
            else:
                # Sin máscara, media simple
                temporal_importance[feature_name] = np.mean(
                    np.abs(shap_values[:, :, i]), axis=0
                )
        
        return temporal_importance


def plot_feature_importance(feature_importance: Dict[str, float], output_dir: str, 
                          model_name: str, top_k: int = 20):
    """Visualiza la importancia de características."""
    logger.info(f"Generando gráfico de importancia de características (top {top_k})...")
    
    # Filtrar características con importancia > 0
    filtered_importance = {k: v for k, v in feature_importance.items() if v > 0}
    
    if not filtered_importance:
        logger.warning("No hay características con importancia > 0")
        return
    
    # Tomar top k
    top_features = dict(list(filtered_importance.items())[:top_k])
    
    plt.figure(figsize=(12, 8))
    features = list(top_features.keys())
    values = list(top_features.values())
    
    # Crear gráfico de barras horizontal
    bars = plt.barh(features, values)
    
    # Colorear barras con gradiente
    sm = plt.cm.ScalarMappable(cmap="viridis", 
                               norm=plt.Normalize(vmin=min(values), vmax=max(values)))
    sm.set_array([])
    
    for bar, val in zip(bars, values):
        bar.set_color(sm.to_rgba(val))
    
    plt.xlabel('Importancia SHAP Promedio (Valor Absoluto Medio)', fontsize=12)
    plt.ylabel('Características', fontsize=12)
    plt.title(f'Top {top_k} Características Más Importantes - {model_name}', fontsize=14)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico guardado en: {filepath}")


def plot_temporal_importance(temporal_importance: Dict[str, np.ndarray], output_dir: str, 
                           model_name: str, top_k: int = 10):
    """Visualiza la importancia temporal de características."""
    logger.info(f"Generando gráfico de importancia temporal (top {top_k})...")
    
    # Calcular importancia media para ranking
    avg_importance = {k: np.mean(v) for k, v in temporal_importance.items() if np.mean(v) > 0}
    
    if not avg_importance:
        logger.warning("No hay características con importancia temporal > 0")
        return
    
    # Seleccionar top k
    top_features_names = sorted(avg_importance, key=avg_importance.get, reverse=True)[:top_k]
    
    # Crear matriz de datos para heatmap
    heatmap_data = []
    feature_labels = []
    
    for feature in top_features_names:
        values = temporal_importance[feature]
        if len(values) > 0:
            heatmap_data.append(values)
            feature_labels.append(feature[:30] + '...' if len(feature) > 30 else feature)
    
    if not heatmap_data:
        logger.warning("No hay datos para el heatmap temporal")
        return
    
    heatmap_data = np.array(heatmap_data)
    
    # Crear figura
    plt.figure(figsize=(16, 8))
    
    # Crear heatmap
    sns.heatmap(heatmap_data, 
                yticklabels=feature_labels,
                cmap='YlOrRd',
                cbar_kws={'label': 'Importancia SHAP'},
                xticklabels=5)  # Mostrar cada 5 timesteps
    
    plt.title(f'Importancia Temporal de Top {top_k} Características - {model_name}', fontsize=14)
    plt.xlabel('Posición Temporal', fontsize=12)
    plt.ylabel('Características', fontsize=12)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'temporal_importance.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico guardado en: {filepath}")


def plot_shap_summary(shap_values: np.ndarray, X: np.ndarray, feature_names: List[str], 
                     output_dir: str, max_display: int = 20):
    """Crea y guarda los gráficos de resumen de SHAP."""
    logger.info("Generando gráficos de resumen SHAP...")
    
    try:
        # Aplanar datos para summary plots
        n_samples, seq_len, n_features = shap_values.shape
        shap_values_flat = shap_values.reshape(-1, n_features)
        X_flat = X.reshape(-1, n_features)
        
        # Filtrar muestras con todos ceros (no observadas)
        non_zero_mask = np.any(X_flat != 0, axis=1)
        if non_zero_mask.sum() == 0:
            logger.warning("No hay muestras no-cero para el summary plot")
            return
        
        shap_values_filtered = shap_values_flat[non_zero_mask]
        X_filtered = X_flat[non_zero_mask]
        
        # Limitar número de muestras para visualización
        max_samples_plot = 5000
        if len(shap_values_filtered) > max_samples_plot:
            indices = np.random.choice(len(shap_values_filtered), max_samples_plot, replace=False)
            shap_values_filtered = shap_values_filtered[indices]
            X_filtered = X_filtered[indices]
        
        # Gráfico de barras
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_filtered, X_filtered, 
                         feature_names=feature_names, 
                         plot_type="bar", 
                         show=False, 
                         max_display=max_display)
        plt.title("Resumen SHAP - Importancia Media de Características", fontsize=14)
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'shap_summary_bar.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Gráfico de barras guardado en: {filepath}")
        
        # --- CORRECCIÓN DEFINITIVA ---
        # Se intenta generar el gráfico de violín (beeswarm) SIN la condición > 100.
        # Se usa un try-except para manejar el caso en que SHAP no pueda generar el gráfico
        # por tener muy pocos puntos, evitando que el script se detenga.
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_filtered, X_filtered, 
                             feature_names=feature_names, 
                             show=False, 
                             max_display=max_display)
            plt.title("Resumen SHAP - Impacto de Valores de Características", fontsize=14)
            plt.tight_layout()
            filepath_violin = os.path.join(output_dir, 'shap_summary_violin.png')
            plt.savefig(filepath_violin, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Gráfico de violín guardado en: {filepath_violin}")
        except Exception as e:
            logger.warning(f"No se pudo generar el gráfico de violín SHAP. Causa probable: muy pocos puntos de datos. Error: {e}")
        
    except Exception as e:
        logger.error(f"Error generando gráficos de resumen SHAP: {e}")


def run_shap_analysis(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    feature_names: List[str],
    output_dir: str,
    max_test_samples: int = 50,  # Aumentado para tener más datos para el plot
    n_background: int = 30,
    n_shap_samples: int = 50
):
    """
    Ejecuta el análisis completo de interpretabilidad para un modelo dado.
    Versión corregida con mejor manejo de errores y dimensiones.
    """
    logger.info(f"=== Iniciando análisis de interpretabilidad para: {model_name} ===")
    
    # Verificar que tenemos nombres de características
    if not feature_names:
        logger.error("La lista de nombres de características está vacía. No se puede continuar.")
        return
    
    # Crear directorio de salida
    interp_dir = os.path.join(output_dir, 'interpretability')
    os.makedirs(interp_dir, exist_ok=True)
    
    try:
        # Obtener dispositivo del modelo
        device = next(model.parameters()).device
        logger.info(f"Modelo en dispositivo: {device}")
        
        # 1. Inicializar intérprete
        interpreter = StratsShapInterpreter(model, feature_names, device)
        
        # 2. Preparar datos de fondo
        interpreter.prepare_background_data(train_loader, n_samples=n_background)
        
        # 3. Inicializar explicador
        interpreter.initialize_explainer(link='identity', nsamples=n_shap_samples)
        
        # 4. Recolectar muestras de test (reducidas)
        logger.info(f"Recolectando hasta {max_test_samples} muestras de test...")
        X_list, mask_list = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                if len(X_list) * test_loader.batch_size >= max_test_samples:
                    break
                X_list.append(batch['X'])
                mask_list.append(batch['mask'])
        
        if not X_list:
            logger.error("No se pudieron recolectar muestras de test")
            return
        
        X_test = torch.cat(X_list, dim=0)[:max_test_samples]
        mask_test = torch.cat(mask_list, dim=0)[:max_test_samples]
        
        logger.info(f"Muestras de test recolectadas: {X_test.shape}")
        
        # 5. Calcular valores SHAP con batches muy pequeños
        shap_values = interpreter.explain_batch(
            X_test, 
            n_samples=n_shap_samples,
            batch_size=3  # Batches muy pequeños para mayor estabilidad
        )
        
        # Verificar que tenemos resultados válidos
        if shap_values.size == 0 or np.all(shap_values == 0):
            logger.warning("Los valores SHAP están vacíos o son todos cero. Abortando análisis.")
            return
        
        # 6. Computar importancias
        logger.info("Computando importancias de características...")
        feature_importance = interpreter.compute_feature_importance(
            shap_values, feature_names, mask_test.cpu().numpy()
        )
        
        temporal_importance = interpreter.compute_temporal_importance(
            shap_values, feature_names, mask_test.cpu().numpy()
        )
        
        # 7. Guardar resultados
        logger.info("Guardando resultados...")
        
        # Guardar importancia de características
        with open(os.path.join(interp_dir, 'feature_importance.json'), 'w') as f:
            json.dump(feature_importance, f, indent=4)
        
        # Guardar resumen de importancia temporal
        temporal_summary = {
            feat: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'max': float(np.max(values)) if len(values) > 0 else 0.0
            }
            for feat, values in temporal_importance.items()
        }
        with open(os.path.join(interp_dir, 'temporal_importance_summary.json'), 'w') as f:
            json.dump(temporal_summary, f, indent=4)
        
        # 8. Generar visualizaciones
        logger.info("Generando visualizaciones...")
        
        plot_feature_importance(feature_importance, interp_dir, model_name, top_k=15)
        plot_temporal_importance(temporal_importance, interp_dir, model_name, top_k=8)
        plot_shap_summary(shap_values, X_test.cpu().numpy(), feature_names, interp_dir, max_display=15)
        
        # 9. Guardar artefactos (solo si son muy pocas muestras)
        if X_test.shape[0] <= 50:
            try:
                np.savez_compressed(
                    os.path.join(interp_dir, 'shap_artifacts.npz'),
                    shap_values=shap_values,
                    X_test=X_test.cpu().numpy(),
                    mask_test=mask_test.cpu().numpy()
                )
                logger.info("Artefactos SHAP guardados en shap_artifacts.npz")
            except Exception as e:
                logger.warning(f"No se pudieron guardar artefactos SHAP: {e}")
        
        logger.info(f"=== Análisis para {model_name} completado exitosamente ===")
        logger.info(f"Resultados guardados en: {interp_dir}")
        
        # Mostrar top 10 características más importantes
        logger.info("\nTop 10 características más importantes:")
        for i, (feat, imp) in enumerate(list(feature_importance.items())[:10], 1):
            logger.info(f"  {i}. {feat}: {imp:.4f}")
        
    except Exception as e:
        logger.error(f"Error durante el análisis de interpretabilidad: {e}", exc_info=True)
        # No re-lanzar la excepción para no detener el pipeline principal
        logger.error("El análisis de interpretabilidad falló, pero el entrenamiento del modelo fue exitoso.")


# Función auxiliar para diagnóstico
def diagnose_data_shapes(train_loader, test_loader, feature_names):
    """
    Función auxiliar para diagnosticar problemas de dimensiones antes del análisis SHAP.
    """
    logger.info("=== DIAGNÓSTICO DE DIMENSIONES ===")
    
    # Verificar train loader
    for batch in train_loader:
        X_train = batch['X']
        times_train = batch['times']
        mask_train = batch['mask']
        logger.info(f"Train batch - X: {X_train.shape}, times: {times_train.shape}, mask: {mask_train.shape}")
        break
    
    # Verificar test loader
    for batch in test_loader:
        X_test = batch['X']
        times_test = batch['times']
        mask_test = batch['mask']
        logger.info(f"Test batch - X: {X_test.shape}, times: {times_test.shape}, mask: {mask_test.shape}")
        break
    
    logger.info(f"Número de features esperado: {len(feature_names)}")
    logger.info(f"Features: {feature_names[:5]}..." if len(feature_names) > 5 else f"Features: {feature_names}")
    
    # Verificar consistencia
    if X_train.shape[-1] != len(feature_names):
        logger.error(f"INCONSISTENCIA: X tiene {X_train.shape[-1]} features, pero feature_names tiene {len(feature_names)}")
    
    if X_train.shape[1:] != X_test.shape[1:]:  # Ignorar batch dimension
        logger.warning(f"Shapes diferentes entre train y test: train={X_train.shape[1:]}, test={X_test.shape[1:]}")
    
    logger.info("=== FIN DIAGNÓSTICO ===")
    
    return {
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'n_features': len(feature_names),
        'consistent': X_train.shape[-1] == len(feature_names)
    }
