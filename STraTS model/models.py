#!/usr/bin/env python3
"""
Módulo para la definición de las arquitecturas de los modelos STraTS, iSTraTS
y los componentes necesarios para el pre-entrenamiento.
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# =============================================================================
# COMPONENTES DE LA ARQUITECTURA
# =============================================================================

class CVE(nn.Module):
    """
    Continuous Value Embedding (CVE).
    """
    def __init__(self, input_dim, hid_dim):
        super().__init__()
        int_dim = int(np.sqrt(hid_dim))
        self.W1 = nn.Parameter(torch.empty(input_dim, int_dim), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.W2 = nn.Parameter(torch.empty(int_dim, hid_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        self.activation = torch.tanh

    def forward(self, x):
        x_trans = torch.matmul(x, self.W1) + self.b1
        x_trans = self.activation(x_trans)
        x_trans = torch.matmul(x_trans, self.W2)
        return x_trans

class FusionAtt(nn.Module):
    """
    Capa de atención para la fusión de representaciones temporales.
    """
    def __init__(self, hid_dim):
        super().__init__()
        self.W = nn.Parameter(torch.empty(hid_dim, hid_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(hid_dim), requires_grad=True)
        self.u = nn.Parameter(torch.empty(hid_dim, 1), requires_grad=True)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.u)
        self.activation = torch.tanh

    def forward(self, x, mask):
        att = torch.matmul(x, self.W) + self.b[None, None, :]
        att = self.activation(att)
        att = torch.matmul(att, self.u)[:, :, 0]
        att = att + (1 - mask) * torch.finfo(att.dtype).min
        att = torch.softmax(att, dim=-1)
        return att

class Transformer(nn.Module):
    """
    Wrapper para el Transformer Encoder de PyTorch.
    """
    def __init__(self, hid_dim, num_layers, num_heads, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim, nhead=num_heads, dim_feedforward=hid_dim * 2,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x, mask):
        attn_mask = ~(mask.bool())
        return self.transformer(x, src_key_padding_mask=attn_mask)

# =============================================================================
# MODELO BASE Y MODELOS DE CLASIFICACIÓN (Fine-tuning)
# =============================================================================

class STraTSModelBase(nn.Module):
    """
    Modelo base que contiene la arquitectura principal de STraTS.
    Genera los embeddings a partir de los tripletes (variable, tiempo, valor).
    """
    def __init__(self, feature_dim, d_model, n_heads, n_layers, dropout, max_seq_length):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        
        self.variable_emb = nn.Embedding(feature_dim, d_model)
        self.cve_time = CVE(1, d_model)
        self.cve_value = CVE(1, d_model)
        
        self.transformer = Transformer(d_model, n_layers, n_heads, dropout)
        self.dropout = dropout

    def get_triplet_embedding(self, X, times, mask):
        var_indices = torch.arange(self.feature_dim).to(X.device)
        variable_emb = self.variable_emb(var_indices).view(1, 1, self.feature_dim, self.d_model)
        time_emb = self.cve_time(times.unsqueeze(-1)).unsqueeze(2)
        value_emb = self.cve_value(X.unsqueeze(-1))
        
        combined_emb = variable_emb + time_emb + value_emb
        masked_emb = combined_emb * mask.unsqueeze(-1)
        aggregated_emb = masked_emb.sum(dim=2)
        
        return aggregated_emb

    def forward(self, X, times, mask):
        """
        El forward del modelo base solo devuelve las representaciones contextuales.
        """
        triplet_emb = self.get_triplet_embedding(X, times, mask)
        triplet_emb = F.dropout(triplet_emb, self.dropout, self.training)
        
        seq_mask = (mask.sum(dim=-1) > 0).float()
        contextual_emb = self.transformer(triplet_emb, seq_mask)
        return contextual_emb

class STraTSModel(STraTSModelBase):
    """
    Modelo STraTS completo para clasificación.
    """
    def __init__(self, feature_dim, d_model, n_heads, n_layers, dropout, max_seq_length, n_classes=2):
        super().__init__(feature_dim, d_model, n_heads, n_layers, dropout, max_seq_length)
        self.fusion_att = FusionAtt(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, X, times, mask):
        seq_mask = (mask.sum(dim=-1) > 0).float()
        contextual_emb = super().forward(X, times, mask)
        
        attention_weights = self.fusion_att(contextual_emb, seq_mask)
        ts_emb = (contextual_emb * attention_weights.unsqueeze(-1)).sum(dim=1)
        
        return self.classifier(ts_emb)

class iSTraTSModel(STraTSModelBase):
    """
    Modelo iSTraTS completo para clasificación.
    """
    def __init__(self, feature_dim, d_model, n_heads, n_layers, dropout, max_seq_length, n_classes=2):
        super().__init__(feature_dim, d_model, n_heads, n_layers, dropout, max_seq_length)
        self.fusion_att = FusionAtt(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, X, times, mask):
        seq_mask = (mask.sum(dim=-1) > 0).float()
        
        triplet_emb = self.get_triplet_embedding(X, times, mask)
        triplet_emb = F.dropout(triplet_emb, self.dropout, self.training)
        
        #_ = self.transformer(triplet_emb, seq_mask) # Se ejecuta pero no se usa para la agregación
        
        attention_weights = self.fusion_att(triplet_emb, seq_mask)
        ts_emb = (triplet_emb * attention_weights.unsqueeze(-1)).sum(dim=1)
        
        return self.classifier(ts_emb)

# =============================================================================
# MODELO PARA PRE-ENTRENAMIENTO (Forecasting)
# =============================================================================

class ForecastingModel(nn.Module):
    """
    Wrapper para añadir una 'cabeza de pronóstico' al modelo base de STraTS.
    """
    def __init__(self, base_model, d_model, feature_dim):
        super().__init__()
        self.base_model = base_model
        self.forecasting_head = nn.Linear(d_model, feature_dim)

    def forward(self, X, times, mask):
        contextual_emb = self.base_model(X, times, mask)
        forecast = self.forecasting_head(contextual_emb)
        return forecast
