#!/usr/bin/env python3
"""
Pipeline unificado para entrenar y evaluar los modelos STraTS e iSTraTS,
con o sin pesos pre-entrenados, análisis de interpretabilidad y visualizaciones completas.
"""
import os
import sys
import argparse
import logging
import json
import pickle
import warnings
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc, f1_score, precision_score
)

# --- Importaciones locales ---
from models import STraTSModel, iSTraTSModel
from run_interpretability import run_shap_analysis
from istrats_contribution import run_istrats_analysis

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# CLASE PARA EARLY STOPPING
# =============================================================================
class EarlyStopping:
    """
    Clase para implementar early stopping. Detiene el entrenamiento cuando una
    métrica monitorizada deja de mejorar tras un número de épocas determinado (paciencia).
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=logging.info, mode='max'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.mode = mode

        if self.mode == 'min':
            self.best_score = np.Inf
        else:
            self.best_score = -np.Inf

    def __call__(self, val_metric, model):
        score = val_metric
        improved = False
        if self.mode == 'max':
            if score > self.best_score + self.delta:
                improved = True
        else: # mode == 'min'
            if score < self.best_score - self.delta:
                improved = True

        if improved:
            if self.verbose:
                self.trace_func(f'Métrica de validación mejorada ({self.best_score:.6f} --> {score:.6f}). Guardando modelo en {self.path}...')
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} de {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        '''Guarda el modelo cuando la métrica de validación mejora.'''
        torch.save(model.state_dict(), self.path)

# =============================================================================
# DATASET Y TRAINER
# =============================================================================
class ClinicalTimeSeriesDataset(Dataset):
    def __init__(self, data_path: str, split: str = 'train'):
        self.data_path = data_path
        self.split = split
        data_file = os.path.join(self.data_path, f'{self.split}_data.pkl')
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        
        self.X = torch.FloatTensor(self.data['X'])
        self.times = torch.FloatTensor(self.data['times'])
        self.masks = torch.FloatTensor(self.data['masks'])
        self.labels = torch.LongTensor(self.data['labels'])
        
        self.feature_names = []
        features_file = os.path.join(self.data_path, 'features.json')
        if os.path.exists(features_file):
            with open(features_file, 'r') as f:
                self.feature_names = json.load(f)
        else:
            logger.warning(f"No se encontró 'features.json' en {self.data_path}.")

        self.patient_ids = self.data.get('patient_ids', [f'Patient_{i}' for i in range(len(self.X))])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {'X': self.X[idx], 'times': self.times[idx], 'mask': self.masks[idx], 'label': self.labels[idx]}

    def get_feature_dim(self):
        return self.X.shape[-1]

    def get_max_seq_length(self):
        return self.X.shape[1]

    def get_class_weights(self):
        counts = torch.bincount(self.labels)
        if len(counts) < 2: return None
        weights = 1. / counts.float()
        return weights / weights.sum()

    def get_feature_names(self) -> List[str]:
        return self.feature_names

class Trainer:
    def __init__(self, model, device, class_weights=None):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
        self.history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 
            'val_auc': [], 'val_auprc': [], 'val_f1': [], 'val_precision': []
        }

    def train_epoch(self, data_loader, optimizer, scheduler):
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for batch in data_loader:
            X, times, mask, labels = batch['X'].to(self.device), batch['times'].to(self.device), batch['mask'].to(self.device), batch['label'].to(self.device)
            optimizer.zero_grad()
            logits = self.model(X, times, mask)
            loss = self.criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)
        if scheduler:
            scheduler.step()
        return total_loss / len(data_loader), total_correct / total_samples

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in data_loader:
                X, times, mask, labels = batch['X'].to(self.device), batch['times'].to(self.device), batch['mask'].to(self.device), batch['label'].to(self.device)
                logits = self.model(X, times, mask)
                loss = self.criterion(logits, labels)
                preds = logits.argmax(1)
                probs = torch.softmax(logits, 1)
                total_loss += loss.item()
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        return (total_loss / len(data_loader), total_correct / total_samples, np.array(all_preds), np.array(all_labels), np.array(all_probs))

    def update_history(self, train_loss, train_acc, val_loss, val_acc, val_auc, val_auprc, val_f1, val_precision):
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['val_auc'].append(val_auc)
        self.history['val_auprc'].append(val_auprc)
        self.history['val_f1'].append(val_f1)
        self.history['val_precision'].append(val_precision)

# =============================================================================
# FUNCIONES DE EVALUACIÓN Y VISUALIZACIÓN
# =============================================================================

def plot_class_distribution(y_labels: np.ndarray, split_name: str, output_dir: str):
    """Genera y guarda un gráfico de barras para la distribución de clases."""
    logger.info(f"Generando gráfico de distribución de clases para el conjunto: {split_name}...")
    unique, counts = np.unique(y_labels, return_counts=True)
    
    class_counts = {0: 0, 1: 0}
    for i in range(len(unique)):
        class_counts[unique[i]] = counts[i]
    
    labels = ['No Evento', 'Evento']
    final_counts = [class_counts[0], class_counts[1]]
    total_samples = sum(final_counts)
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=labels, y=final_counts, palette=['skyblue', 'salmon'])
    plt.title(f'Distribución de Clases - {split_name.capitalize()}', fontsize=16)
    plt.xlabel('Clase', fontsize=12)
    plt.ylabel('Número de Muestras', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, count in enumerate(final_counts):
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        ax.text(i, count, f'{count}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3'))

    # Ampliar el eje Y para que no se corten las etiquetas
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'class_distribution_{split_name}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico de distribución de clases para '{split_name}' guardado en: {filepath}")

def plot_training_history(history: Dict[str, List[float]], output_dir: str):
    """Visualiza el historial de entrenamiento."""
    logger.info("Generando gráfico del historial de entrenamiento...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Historial de Entrenamiento y Validación', fontsize=16)
    
    metrics = [
        ('Loss', 'loss', 'train_loss', 'val_loss'),
        ('Accuracy', 'acc', 'train_acc', 'val_acc'),
        ('AUC', 'auc', None, 'val_auc'),
        ('AUPRC', 'auprc', None, 'val_auprc'),
        ('F1-score', 'f1', None, 'val_f1'),
        ('Precision', 'precision', None, 'val_precision')
    ]
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for ax, (title, key, train_key, val_key) in zip(axes.flatten(), metrics):
        if train_key and train_key in history:
            ax.plot(history[train_key], label=f'Train {title}', color=colors[0], linewidth=2)
        if val_key in history:
            ax.plot(history[val_key], label=f'Validation {title}', color=colors[1], linewidth=2)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Época')
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(output_dir, 'training_history.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Historial de entrenamiento guardado en: {filepath}")

def plot_confusion_matrix_and_report(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str, split_name: str, class_names: List[str] = None):
    """Genera y guarda la matriz de confusión y el reporte de clasificación."""
    logger.info(f"Generando matriz de confusión para {split_name}...")
    if class_names is None:
        class_names = ['No Evento', 'Evento']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14})
    plt.title(f'Matriz de Confusión - {split_name.capitalize()}', fontsize=16)
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Etiqueta Verdadera', fontsize=12)
    plt.tight_layout()
    cm_filepath = os.path.join(output_dir, f'confusion_matrix_{split_name}.png')
    plt.savefig(cm_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Matriz de confusión guardada en: {cm_filepath}")

    report_filepath = os.path.join(output_dir, f'classification_report_{split_name}.txt')
    with open(report_filepath, 'w') as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))
    logger.info(f"Reporte de clasificación guardado en: {report_filepath}")

def plot_roc_and_pr_curves(y_true: np.ndarray, y_probs: np.ndarray, output_dir: str, split_name: str, model_name: str = 'STraTS'):
    """Genera y guarda las curvas ROC y Precision-Recall en un solo gráfico."""
    logger.info(f"Generando curvas ROC y P-R para {split_name}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Curva ROC (ax1) ---
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador Aleatorio')
    ax1.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    ax1.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
    ax1.set_title(f'Curva ROC - {split_name.capitalize()}', fontsize=14)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # --- Curva Precision-Recall (ax2) ---
    precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
    pr_auc = average_precision_score(y_true, y_probs[:, 1])
    positive_class_ratio = np.sum(y_true == 1) / len(y_true)
    ax2.plot(recall, precision, color='b', lw=2, label=f'Curva P-R (AUPRC = {pr_auc:.3f})')
    ax2.axhline(y=positive_class_ratio, color='r', linestyle='--', label=f'Línea Base (Azar) (P={positive_class_ratio:.3f})')
    ax2.set_xlabel('Recall (Sensibilidad)')
    ax2.set_ylabel('Precision (Valor Predictivo Positivo)')
    ax2.set_title(f'Curva Precision-Recall - {split_name.capitalize()}', fontsize=14)
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'roc_pr_curves_{split_name}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Curvas ROC y P-R guardadas en: {filepath}")

def plot_evaluation_summary(summary_data: Dict[str, Dict[str, float]], output_dir: str):
    """
    Genera un gráfico de líneas y uno de barras resumiendo las métricas finales
    para los conjuntos de train, val y test.
    """
    logger.info("Generando gráficos de resumen de evaluación final...")
    df = pd.DataFrame(summary_data).T
    df.index.name = 'Conjunto'
    df = df.reset_index()
    
    plt.figure(figsize=(12, 7))
    for metric in df.columns:
        if metric != 'Conjunto':
            plt.plot(df['Conjunto'], df[metric], marker='o', linestyle='-', label=metric)
    
    plt.title('Resumen de Métricas de Evaluación por Conjunto', fontsize=16)
    plt.xlabel('Conjunto de Datos', fontsize=12)
    plt.ylabel('Puntuación', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Métricas', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    filepath_line = os.path.join(output_dir, 'evaluation_summary_lineplot.png')
    plt.savefig(filepath_line, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico de resumen (líneas) guardado en: {filepath_line}")

    df_melted = df.melt(id_vars='Conjunto', var_name='Métrica', value_name='Puntuación')
    plt.figure(figsize=(15, 8))
    ax = sns.barplot(data=df_melted, x='Métrica', y='Puntuación', hue='Conjunto', palette='viridis')
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=9)

    plt.title('Resumen de Métricas de Evaluación por Conjunto', fontsize=16)
    plt.xlabel('Métrica', fontsize=12)
    plt.ylabel('Puntuación', fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(title='Conjunto')
    plt.tight_layout()
    filepath_bar = os.path.join(output_dir, 'evaluation_summary_barplot.png')
    plt.savefig(filepath_bar, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico de resumen (barras) guardado en: {filepath_bar}")

def plot_metrics_heatmap(summary_data: Dict[str, Dict[str, float]], output_dir: str):
    """Genera un heatmap con el resumen de las métricas finales."""
    logger.info("Generando heatmap de resumen de métricas...")
    # Reordenar columnas para que coincida con la imagen de ejemplo
    df = pd.DataFrame(summary_data)[['test', 'train', 'val']]
    # Reordenar filas
    df = df.reindex(['Accuracy', 'AUC', 'AUPRC', 'F1-score', 'Precision'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlOrRd", linewidths=.5, cbar_kws={'label': 'Valor de la Métrica'})
    plt.title('Resumen de Métricas por Conjunto de Datos', fontsize=16)
    plt.xlabel('Conjunto', fontsize=12)
    plt.ylabel('Métrica', fontsize=12)
    plt.yticks(rotation=0) 
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'evaluation_summary_heatmap.png')
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info(f"Heatmap de resumen guardado en: {filepath}")

# =============================================================================
# BUCLE DE ENTRENAMIENTO PRINCIPAL
# =============================================================================
def run_experiment(model_type: str, config: Dict[str, Any], output_dir: str, data_path: str, pretrained_path: str = None):
    """
    Función principal para ejecutar el experimento de entrenamiento y evaluación.
    """
    logger.info(f"--- Iniciando experimento para {model_type.upper()} ---")
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Usando dispositivo: {device}")

    loaders = {}
    datasets = {}
    max_seq_len = 0
    for split in ['train', 'val', 'test']:
        try:
            dataset = ClinicalTimeSeriesDataset(data_path=data_path, split=split)
            datasets[split] = dataset
            batch_size = config['training']['batch_size']
            loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
            
            plot_class_distribution(dataset.labels.numpy(), split, output_dir)

            if split == 'train':
                feature_names = dataset.get_feature_names()
                class_weights = dataset.get_class_weights()
                input_dim = dataset.get_feature_dim()
                max_seq_len = dataset.get_max_seq_length()
        except FileNotFoundError:
            logger.error(f"Error: Archivo de datos para '{split}' no encontrado.")
            return

    model_config = config['model']
    model_args = {
        "feature_dim": input_dim, "d_model": model_config['d_model'], 
        "n_classes": model_config['n_classes'], "n_heads": model_config['n_heads'],
        "n_layers": model_config['n_layers'], "dropout": model_config['dropout'],
        "max_seq_length": max_seq_len
    }
    
    if model_type == 'strats':
        model = STraTSModel(**model_args)
    elif model_type == 'istrats':
        model = iSTraTSModel(**model_args)
    else:
        logger.error(f"Tipo de modelo '{model_type}' no soportado.")
        return

    if pretrained_path:
        try:
            state_dict = torch.load(pretrained_path, map_location=device)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info(f"Pesos base cargados desde {pretrained_path}")
            if missing:
                logger.info(f"Capas inicializadas desde cero (no estaban en el preentrenamiento): {missing}")
            if unexpected:
                logger.warning(f"Pesos inesperados en el checkpoint (no se usaron): {unexpected}")
        except Exception as e:
            logger.error(f"Error al cargar los pesos pre-entrenados: {e}")
            return

    training_config = config['training']
    trainer = Trainer(model, device, class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config['learning_rate'], weight_decay=training_config['weight_decay'])
    scheduler = None
    early_stopping = EarlyStopping(patience=training_config.get('patience', 10), verbose=True, path=os.path.join(output_dir, 'checkpoint.pt'))

    for epoch in range(1, training_config['num_epochs'] + 1):
        train_loss, train_acc = trainer.train_epoch(loaders['train'], optimizer, scheduler)
        val_loss, val_acc, val_preds, val_labels, val_probs = trainer.evaluate(loaders['val'])
        
        val_auc = roc_auc_score(val_labels, val_probs[:, 1])
        val_auprc = average_precision_score(val_labels, val_probs[:, 1])
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_precision = precision_score(val_labels, val_preds, zero_division=0)

        trainer.update_history(train_loss, train_acc, val_loss, val_acc, val_auc, val_auprc, val_f1, val_precision)
        
        logger.info(f"Época {epoch}/{training_config['num_epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}")
        
        early_stopping(val_auc, model)
        if early_stopping.early_stop:
            logger.info("Early stopping activado.")
            break

    model.load_state_dict(torch.load(os.path.join(output_dir, 'checkpoint.pt'), map_location=device))
    logger.info("Mejor modelo cargado para la evaluación final.")
    final_model_path = os.path.join(output_dir, 'best_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Modelo final guardado para producción en: {final_model_path}")
    
    evaluation_summary = {}
    logger.info("\n--- Iniciando Evaluación Final en todos los conjuntos ---")
    for split in ['train', 'val', 'test']:
        loss, acc, preds, labels, probs = trainer.evaluate(loaders[split])
        
        auc_score = roc_auc_score(labels, probs[:, 1])
        auprc_score = average_precision_score(labels, probs[:, 1])
        f1 = f1_score(labels, preds, zero_division=0)
        precision = precision_score(labels, preds, zero_division=0)
        
        evaluation_summary[split] = {
            'Accuracy': acc, 'AUC': auc_score, 'AUPRC': auprc_score,
            'F1-score': f1, 'Precision': precision
        }
        
        logger.info(f"--- Métricas en conjunto '{split.upper()}' ---")
        logger.info(f"  Loss: {loss:.4f} | Accuracy: {acc:.4f} | AUC: {auc_score:.4f} | AUPRC: {auprc_score:.4f} | F1: {f1:.4f} | Precision: {precision:.4f}")

        plot_confusion_matrix_and_report(labels, preds, output_dir, split)
        plot_roc_and_pr_curves(labels, probs, output_dir, split, model_type.upper())

    plot_training_history(trainer.history, output_dir)
    plot_evaluation_summary(evaluation_summary, output_dir)
    plot_metrics_heatmap(evaluation_summary, output_dir)
    
    # --- AÑADIDO: Guardar el resumen de la evaluación en un archivo JSON ---
    summary_filepath = os.path.join(output_dir, 'evaluation_summary.json')
    try:
        with open(summary_filepath, 'w') as f:
            json.dump(evaluation_summary, f, indent=4)
        logger.info(f"Resumen de la evaluación guardado en: {summary_filepath}")
    except Exception as e:
        logger.error(f"No se pudo guardar el resumen de la evaluación: {e}")
    # --------------------------------------------------------------------



    logger.info("\n--- Iniciando Análisis de Interpretabilidad ---")
    if model_type == 'istrats':
        try:
            run_istrats_analysis(model=model, test_loader=loaders['test'], feature_names=feature_names, device=device, output_dir=output_dir)
        except Exception as e:
            logger.error(f"Falló el análisis de contribution scores para iSTraTS: {e}", exc_info=True)
    
    try:
        run_shap_analysis(model=model, model_name=f"{model_type.upper()}", train_loader=loaders['train'], test_loader=loaders['test'], feature_names=feature_names, output_dir=output_dir)
    except Exception as e:
        logger.error(f"Falló el análisis de interpretabilidad SHAP: {e}", exc_info=True)

    logger.info(f"--- Experimento para {model_type.upper()} finalizado ---")
    return evaluation_summary 

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Entrenamiento para STraTS e iSTraTS")
    parser.add_argument('--model_type', type=str, required=True, choices=['strats', 'istrats'])
    parser.add_argument('--config', type=str, default='strats_config.json')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='strats_data')
    parser.add_argument('--pretrained_path', type=str, default=None)
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: Archivo de configuración '{args.config}' no encontrado.")
        return
    except json.JSONDecodeError:
        logger.error(f"Error: Archivo de configuración '{args.config}' no es un JSON válido.")
        return

    run_experiment(
        model_type=args.model_type,
        config=config,
        output_dir=args.output_dir,
        data_path=args.data_path,
        pretrained_path=args.pretrained_path
    )

if __name__ == '__main__':
    main()
