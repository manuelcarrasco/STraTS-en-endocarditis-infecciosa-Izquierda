#!/usr/bin/env python3
"""
Script para realizar el pre-entrenamiento auto-supervisado de STraTS.
"""
import os
import sys
import argparse
import logging
import json
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- Importaciones locales ---
from models import STraTSModelBase, ForecastingModel
# Asumimos que la clase del dataset está en un fichero `data_utils.py`
# para mayor orden. Si no, puedes importarla desde `train.py`.
from train import ClinicalTimeSeriesDataset 

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("pretrain.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

class Pretrainer:
    """
    Clase para manejar el bucle de pre-entrenamiento auto-supervisado.
    """
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss(reduction='none')

    def train_epoch(self, data_loader, optimizer, scheduler, context_frac=0.8):
        self.model.train()
        total_loss = 0
        
        for batch in data_loader:
            X, times, mask = batch['X'].to(self.device), batch['times'].to(self.device), batch['mask'].to(self.device)
            optimizer.zero_grad()
            
            seq_len = X.shape[1]
            context_len = int(seq_len * context_frac)
            horizon_len = seq_len - context_len

            if context_len == 0 or horizon_len == 0:
                continue

            forecast = self.model(X, times, mask)
            
            forecast_horizon = forecast[:, context_len:, :]
            true_horizon = X[:, context_len:, :]
            horizon_mask = mask[:, context_len:, :]

            loss = self.criterion(forecast_horizon, true_horizon)
            masked_loss = (loss * horizon_mask).sum() / (horizon_mask.sum() + 1e-8)
            
            masked_loss.backward()
            optimizer.step()
            
            total_loss += masked_loss.item()
        
        if scheduler:
            scheduler.step()
            
        return total_loss / len(data_loader)

def run_pretraining(config, output_dir, epochs):
    logger.info("--- Iniciando Pre-entrenamiento Auto-Supervisado de STraTS ---")
    os.makedirs(output_dir, exist_ok=True)

    data_path = 'strats_data'
    if not os.path.exists(data_path):
        logger.error(f"Directorio de datos '{data_path}' no encontrado.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Usando dispositivo: {device}")

    full_dataset = ClinicalTimeSeriesDataset(data_path, 'train')
    data_loader = DataLoader(
        full_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    feature_dim = full_dataset.get_feature_dim()
    max_seq_len = full_dataset.get_max_seq_length()
    
    base_model = STraTSModelBase(
        feature_dim=feature_dim,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        dropout=config['model']['dropout'],
        max_seq_length=max_seq_len
    )
    model = ForecastingModel(base_model, config['model']['d_model'], feature_dim)
    
    logger.info(f"Modelo de pronóstico creado con {sum(p.numel() for p in model.parameters()):,} parámetros.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    pretrainer = Pretrainer(model, device)

    logger.info(f"Iniciando pre-entrenamiento por {epochs} épocas...")
    for epoch in range(epochs):
        train_loss = pretrainer.train_epoch(data_loader, optimizer, scheduler)
        logger.info(f"Época de Pre-entrenamiento {epoch+1:03d}/{epochs:03d} | Loss (MSE): {train_loss:.6f}")

    output_path = os.path.join(output_dir, 'pretrained_strats_body.pth')
    torch.save(model.base_model.state_dict(), output_path)
    
    logger.info(f"--- Pre-entrenamiento finalizado ---")
    logger.info(f"Pesos del modelo pre-entrenado guardados en: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Pre-entrenamiento para STraTS")
    parser.add_argument('--config', type=str, default='strats_config.json')
    parser.add_argument('--output_dir', type=str, default='results/pretrain')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    run_pretraining(config, args.output_dir, args.epochs)

if __name__ == '__main__':
    main()
