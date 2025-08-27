import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Optional
import warnings
import re
import json
import logging
warnings.filterwarnings('ignore')

class STraTSDataAdapter:
    """
    Adaptador mejorado para convertir datos clínicos en formato tripletas al formato requerido por STraTS
    Incluye unificación de nombres de variables con prefijos temporales
    """
    
    def __init__(self, tripletas_dir: str = 'tripletas', output_dir: str = 'strats_data'):
        self.tripletas_dir = tripletas_dir
        self.output_dir = output_dir
        self.feature_encoder = LabelEncoder()
        self.scalers = {}
        self.feature_to_idx = {}
        self.idx_to_feature = {}
        self.max_seq_length = 0
        
        # Definir patrones de prefijos temporales a eliminar
        self.temporal_prefixes = [
            r'^ING\s+',      # ING Insuficiencia cardiaca -> Insuficiencia cardiaca
            r'^S\d+\s+',     # S1 Insuficiencia cardiaca -> Insuficiencia cardiaca
            r'^SEM\d+\s+',   # SEM1 Variable -> Variable
            r'^T\d+\s+',     # T1 Variable -> Variable
            r'^WEEK\d+\s+',  # WEEK1 Variable -> Variable
            r'^DIA\d+\s+',   # DIA1 Variable -> Variable
            r'^D\d+\s+',     # D1 Variable -> Variable
        ]
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
    
    def unify_parameter_name(self, parameter_name: str) -> str:
        """
        Unificar nombres de parámetros eliminando prefijos temporales
        """
        if pd.isna(parameter_name) or not isinstance(parameter_name, str):
            return parameter_name
        
        unified_name = parameter_name.strip()
        
        # Aplicar cada patrón de prefijo temporal
        for prefix_pattern in self.temporal_prefixes:
            unified_name = re.sub(prefix_pattern, '', unified_name, flags=re.IGNORECASE)
        
        # Limpiar espacios extra
        unified_name = re.sub(r'\s+', ' ', unified_name).strip()
        
        return unified_name
    
    def load_tripletas_data(self) -> Dict:
        """
        Cargar todos los archivos de tripletas y organizarlos por paciente
        Incluye unificación de nombres de parámetros
        """
        print("Cargando datos de tripletas...")
        patients_data = {}
        
        # Para tracking de unificaciones
        original_to_unified = {}
        
        for filename in os.listdir(self.tripletas_dir):
            if filename.endswith('.txt'):
                patient_id = filename.replace('.txt', '')
                filepath = os.path.join(self.tripletas_dir, filename)
                
                try:
                    df = pd.read_csv(filepath)
                    # Validar estructura
                    if len(df) > 0 and all(col in df.columns for col in ['Time', 'Parameter', 'Value']):
                        # Unificar nombres de parámetros
                        df['Parameter_Original'] = df['Parameter'].copy()  # Guardar original para debug
                        df['Parameter'] = df['Parameter'].apply(self.unify_parameter_name)
                        
                        # Track unificaciones para reporte
                        for orig, unified in zip(df['Parameter_Original'], df['Parameter']):
                            if orig != unified:
                                if orig not in original_to_unified:
                                    original_to_unified[orig] = unified
                        
                        # Filtrar valores NaN explícitamente
                        df = df.dropna(subset=['Value'])
                        
                        # Eliminar filas donde Parameter quedó vacío
                        df = df[df['Parameter'].str.strip() != '']
                        
                        if len(df) > 0:
                            # Eliminar columna temporal de debug
                            df = df.drop('Parameter_Original', axis=1)
                            patients_data[patient_id] = df
                        else:
                            print(f"Advertencia: {filename} no tiene valores válidos después de procesar")
                    else:
                        print(f"Advertencia: {filename} no tiene la estructura correcta")
                except Exception as e:
                    print(f"Error cargando {filename}: {e}")
                    continue
        
        # Mostrar reporte de unificaciones
        if original_to_unified:
            print(f"\n=== Unificaciones realizadas ({len(original_to_unified)} cambios) ===")
            for orig, unified in sorted(original_to_unified.items()):
                print(f"  '{orig}' -> '{unified}'")
            print("=" * 50)
        
        print(f"Cargados {len(patients_data)} pacientes")
        return patients_data
    
    def analyze_data_structure(self, patients_data: Dict) -> Dict:
        """
        Analizar la estructura de los datos para entender patrones temporales
        """
        print("Analizando estructura de datos...")
        
        all_features = set()
        all_times = set()
        demographic_features = set()
        temporal_features = set()
        feature_types = {}
        
        for patient_id, df in patients_data.items():
            for _, row in df.iterrows():
                time = row['Time']
                parameter = row['Parameter']
                value = row['Value']
                
                all_features.add(parameter)
                all_times.add(time)
                
                # Clasificar por tiempo
                if time == 0:
                    demographic_features.add(parameter)
                else:
                    temporal_features.add(parameter)
                
                # Analizar tipo de dato
                if parameter not in feature_types:
                    feature_types[parameter] = {'numeric': 0, 'categorical': 0}
                
                try:
                    float(value)
                    feature_types[parameter]['numeric'] += 1
                except (ValueError, TypeError):
                    feature_types[parameter]['categorical'] += 1
        
        analysis = {
            'total_features': len(all_features),
            'total_unique_times': len(all_times),
            'demographic_features': list(demographic_features),
            'temporal_features': list(temporal_features),
            'all_times_sorted': sorted(list(all_times)),
            'feature_types': feature_types
        }
        
        print(f"Features demográficas: {len(demographic_features)}")
        print(f"Features temporales: {len(temporal_features)}")
        print(f"Tiempos únicos: {analysis['all_times_sorted']}")
        
        # Mostrar features más comunes
        feature_counts = {}
        for df in patients_data.values():
            for param in df['Parameter']:
                feature_counts[param] = feature_counts.get(param, 0) + 1
        
        print(f"\nTop 10 features más frecuentes:")
        for feat, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {feat}: {count} ocurrencias")
        
        return analysis
    
    def create_feature_encoding(self, patients_data: Dict):
        """
        Crear mapeo de features a índices numéricos
        """
        all_features = set()
        for df in patients_data.values():
            all_features.update(df['Parameter'].unique())
        
        all_features = sorted(list(all_features))
        self.feature_to_idx = {feat: idx for idx, feat in enumerate(all_features)}
        self.idx_to_feature = {idx: feat for feat, idx in self.feature_to_idx.items()}
        
        print(f"Creados mapeos para {len(all_features)} features")
    
    def normalize_time_stamps(self, patients_data: Dict, time_unit: str = 'weeks') -> Dict:
        """
        Normalizar timestamps considerando que algunos son días y otros semanas
        """
        print(f"Normalizando timestamps a {time_unit}...")
        
        normalized_data = {}
        
        for patient_id, df in patients_data.items():
            df_norm = df.copy()
            
            for idx, row in df_norm.iterrows():
                parameter = row['Parameter']
                time = row['Time']
                
                # Convertir días a semanas para variables específicas
                if parameter in ['tiempo_hasta_cirugia', 't_muerte_alta', 'Cirugia', 'Muerte_hospitalaria']:
                    if time_unit == 'weeks' and time > 0:
                        df_norm.at[idx, 'Time'] = time / 7.0
                
            normalized_data[patient_id] = df_norm
        
        return normalized_data
    
    def create_sequences(self, patients_data: Dict, max_seq_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Convertir datos de tripletas a secuencias para STraTS - SIN generar NaN artificiales
        """
        print("Creando secuencias temporales...")
        
        n_features = len(self.feature_to_idx)
        patient_ids = list(patients_data.keys())
        n_patients = len(patient_ids)
        
        # Determinar longitud máxima de secuencia
        if max_seq_length is None:
            max_times = []
            for df in patients_data.values():
                unique_times = sorted(df['Time'].unique())
                max_times.append(len(unique_times))
            max_seq_length = max(max_times)
        
        self.max_seq_length = max_seq_length
        print(f"Longitud máxima de secuencia: {max_seq_length}")
        
        # Inicializar arrays - CLAVE: usar 0.0 en lugar de NaN
        X = np.zeros((n_patients, max_seq_length, n_features), dtype=np.float32)
        times = np.zeros((n_patients, max_seq_length), dtype=np.float32)
        masks = np.zeros((n_patients, max_seq_length, n_features), dtype=np.float32)
        
        for patient_idx, patient_id in enumerate(patient_ids):
            df = patients_data[patient_id]
            
            # Obtener timestamps únicos ordenados para este paciente
            unique_times = sorted(df['Time'].unique())
            
            for time_idx, time in enumerate(unique_times[:max_seq_length]):
                times[patient_idx, time_idx] = time
                
                # Obtener todos los valores para este timestamp
                time_data = df[df['Time'] == time]
                
                for _, row in time_data.iterrows():
                    parameter = row['Parameter']
                    value = row['Value']
                    
                    if parameter in self.feature_to_idx:
                        feat_idx = self.feature_to_idx[parameter]
                        
                        try:
                            # Intentar convertir a numérico
                            numeric_value = float(value)
                            X[patient_idx, time_idx, feat_idx] = numeric_value
                            masks[patient_idx, time_idx, feat_idx] = 1.0
                        except (ValueError, TypeError):
                            # Para valores categóricos, usar encoding
                            if isinstance(value, str) and value.strip():
                                # Usar hash más estable para strings
                                encoded_value = abs(hash(value.lower().strip())) % 1000
                                X[patient_idx, time_idx, feat_idx] = encoded_value
                                masks[patient_idx, time_idx, feat_idx] = 1.0
        
        print(f"Secuencias creadas: {X.shape}")
        print(f"Valores observados: {masks.sum():,.0f} / {masks.size:,.0f} ({masks.sum()/masks.size*100:.2f}%)")
        
        # Verificar que no hay NaN
        if np.isnan(X).any():
            print("ADVERTENCIA: Se encontraron valores NaN en X - esto no debería ocurrir")
            nan_count = np.isnan(X).sum()
            print(f"Número de NaN: {nan_count}")
        else:
            print("✓ No hay valores NaN en las secuencias")
        
        return X, times, masks, patient_ids
    
    def _determine_feature_types(self, patients_data: Dict) -> Dict[str, str]:
        """
        Determinar si cada feature es numérica o categórica basado en los datos originales
        """
        feature_types = {}
        
        # Analizar cada feature en todos los pacientes
        for patient_id, df in patients_data.items():
            for _, row in df.iterrows():
                parameter = row['Parameter']
                value = row['Value']
                
                if parameter not in feature_types:
                    # Intentar determinar tipo basado en múltiples valores
                    numeric_count = 0
                    categorical_count = 0
                    total_values = 0
                    
                    # Analizar todos los valores de este parámetro en todos los pacientes
                    for pid, patient_df in patients_data.items():
                        param_values = patient_df[patient_df['Parameter'] == parameter]['Value']
                        
                        for val in param_values:
                            total_values += 1
                            try:
                                float_val = float(val)
                                # Verificar si es realmente numérico o solo un código
                                # Heurística: si es entero pequeño y hay pocos valores únicos, 
                                # podría ser categórico
                                if float_val.is_integer() and 0 <= float_val <= 10:
                                    # Verificar diversidad de valores para este parámetro
                                    unique_vals = len(patient_df[patient_df['Parameter'] == parameter]['Value'].unique())
                                    if unique_vals <= 5:  # Pocos valores únicos sugiere categórico
                                        categorical_count += 1
                                    else:
                                        numeric_count += 1
                                else:
                                    numeric_count += 1
                            except (ValueError, TypeError):
                                categorical_count += 1
                    
                    # Decidir tipo basado en mayoría
                    if total_values > 0:
                        if numeric_count > categorical_count:
                            feature_types[parameter] = 'numeric'
                        else:
                            feature_types[parameter] = 'categorical'
                    else:
                        feature_types[parameter] = 'categorical'  # Default
        
        # Mostrar clasificación para debug
        numeric_features = [f for f, t in feature_types.items() if t == 'numeric']
        categorical_features = [f for f, t in feature_types.items() if t == 'categorical']
        
        print(f"\n=== Clasificación de features ===")
        print(f"Numéricas ({len(numeric_features)}): {numeric_features[:10]}{'...' if len(numeric_features) > 10 else ''}")
        print(f"Categóricas ({len(categorical_features)}): {categorical_features[:10]}{'...' if len(categorical_features) > 10 else ''}")
        print("=" * 40)
        
        return feature_types
    
    def normalize_features(self, X: np.ndarray, masks: np.ndarray, patients_data: Dict) -> np.ndarray:
        """
        Normalizar SOLO features numéricas usando StandardScaler
        Las features categóricas se mantienen sin normalizar
        """
        print("Normalizando features numéricas...")
        
        X_normalized = X.copy()
        n_features = X.shape[2]
        features_normalized = 0
        features_skipped_categorical = 0
        
        # Determinar qué features son numéricas vs categóricas
        feature_types = self._determine_feature_types(patients_data)
        
        for feat_idx in range(n_features):
            feature_name = self.idx_to_feature[feat_idx]
            
            # Solo normalizar si es numérica
            if feature_types.get(feature_name, 'categorical') == 'numeric':
                # Extraer todos los valores observados para esta feature (mask = 1)
                feature_mask = masks[:, :, feat_idx] == 1
                
                if feature_mask.sum() > 1:  # Si hay al menos 2 valores observados
                    feature_values = X[:, :, feat_idx][feature_mask]
                    
                    if len(feature_values) > 1:
                        # Crear y ajustar scaler
                        scaler = StandardScaler()
                        scaler.fit(feature_values.reshape(-1, 1))
                        
                        # Normalizar valores observados
                        normalized_values = scaler.transform(feature_values.reshape(-1, 1)).flatten()
                        X_normalized[:, :, feat_idx][feature_mask] = normalized_values
                        
                        # Guardar scaler
                        self.scalers[feature_name] = scaler
                        features_normalized += 1
                    else:
                        print(f"Advertencia: Feature numérica '{feature_name}' tiene muy pocos valores para normalizar")
            else:
                # Feature categórica - no normalizar
                features_skipped_categorical += 1
                print(f"Saltando normalización para feature categórica: '{feature_name}'")
        
        print(f"Features numéricas normalizadas: {features_normalized}")
        print(f"Features categóricas sin normalizar: {features_skipped_categorical}")
        print(f"Total features: {n_features}")
        return X_normalized
    
    def create_labels_from_outcomes(self, patients_data: Dict, patient_ids: List[str]) -> np.ndarray:
        """
        Crear etiquetas de clasificación basadas en outcomes clínicos
        NOTA: Esta función usa los datos ORIGINALES antes de filtrar outcomes
        """
        print("Creando etiquetas de clasificación...")
        
        # Necesitamos recargar los datos originales para obtener las etiquetas
        original_patients_data = {}
        for filename in os.listdir(self.tripletas_dir):
            if filename.endswith('.txt'):
                patient_id = filename.replace('.txt', '')
                if patient_id in patient_ids:  # Solo para pacientes que tenemos
                    filepath = os.path.join(self.tripletas_dir, filename)
                    try:
                        df = pd.read_csv(filepath)
                        # Aplicar unificación de nombres también aquí
                        df['Parameter'] = df['Parameter'].apply(self.unify_parameter_name)
                        original_patients_data[patient_id] = df
                    except Exception as e:
                        print(f"Error cargando {filename} para etiquetas: {e}")
        
        labels = np.zeros(len(patient_ids))
        
        for idx, patient_id in enumerate(patient_ids):
            if patient_id in original_patients_data:
                df = original_patients_data[patient_id]
                
                # Buscar outcomes de interés (muerte hospitalaria)
                death_data = df[df['Parameter'] == 'Muerte_hospitalaria']
                if not death_data.empty:
                    death_value = death_data.iloc[0]['Value']
                    try:
                        labels[idx] = float(death_value)
                    except (ValueError, TypeError):
                        labels[idx] = 1 if str(death_value).lower() in ['true', '1', 'yes', 'sí'] else 0
        
        print(f"Distribución de etiquetas: {np.bincount(labels.astype(int))}")
        return labels
    
    def save_strats_format(self, X: np.ndarray, times: np.ndarray, masks: np.ndarray, 
                          labels: np.ndarray, patient_ids: List[str], split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        """
        Guardar datos en formato compatible con STraTS
        """
        print("Guardando datos en formato STraTS...")
        
        n_patients = len(patient_ids)
        np.random.seed(42)
        indices = np.random.permutation(n_patients)
        
        # Dividir en train/val/test
        train_end = int(split_ratios[0] * n_patients)
        val_end = train_end + int(split_ratios[1] * n_patients)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        # Crear splits
        splits = {
            'train': {
                'X': X[train_idx],
                'times': times[train_idx],
                'masks': masks[train_idx],
                'labels': labels[train_idx],
                'patient_ids': [patient_ids[i] for i in train_idx]
            },
            'val': {
                'X': X[val_idx],
                'times': times[val_idx],
                'masks': masks[val_idx],
                'labels': labels[val_idx],
                'patient_ids': [patient_ids[i] for i in val_idx]
            },
            'test': {
                'X': X[test_idx],
                'times': times[test_idx],
                'masks': masks[test_idx],
                'labels': labels[test_idx],
                'patient_ids': [patient_ids[i] for i in test_idx]
            }
        }
        
        # Guardar cada split
        for split_name, split_data in splits.items():
            filename = os.path.join(self.output_dir, f'{split_name}_data.pkl')
            with open(filename, 'wb') as f:
                pickle.dump(split_data, f)
            print(f"Guardado {split_name}: {len(split_data['patient_ids'])} pacientes")
        
        # Guardar metadatos
        metadata = {
            'feature_to_idx': self.feature_to_idx,
            'idx_to_feature': self.idx_to_feature,
            'scalers': self.scalers,
            'n_features': len(self.feature_to_idx),
            'max_seq_length': self.max_seq_length,
            'data_shape': X.shape
        }
        
        with open(os.path.join(self.output_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Metadatos guardados. Total features: {metadata['n_features']}")
        return splits
    
    def process_complete_pipeline(self, max_seq_length: Optional[int] = None, 
                                time_unit: str = 'weeks', split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        """
        Ejecutar el pipeline completo de procesamiento
        """
        print("=== Iniciando pipeline de procesamiento para STraTS ===")
        
        # 1. Cargar datos (ya incluye unificación de nombres)
        patients_data = self.load_tripletas_data()
        if not patients_data:
            raise ValueError("No se pudieron cargar datos de tripletas")
        
        # 2. Analizar estructura
        analysis = self.analyze_data_structure(patients_data)
        
        # 3. Normalizar timestamps
        patients_data = self.normalize_time_stamps(patients_data, time_unit)

        # 4. Filtrar las variables de resultado de los datos de entrada (DESPUÉS de obtener etiquetas)
        outcome_vars_to_exclude = ['Muerte_hospitalaria', 't_muerte_alta','tiempo_hasta_cirugia','num_protocolo']
        print(f"Excluyendo variables de resultado: {outcome_vars_to_exclude}")
        
        # Primero obtener las etiquetas con datos completos
        patient_ids_temp = list(patients_data.keys())
        labels = self.create_labels_from_outcomes(patients_data, patient_ids_temp)
        
        # Luego filtrar las variables de resultado
        for patient_id in patients_data:
            df = patients_data[patient_id]
            patients_data[patient_id] = df[~df['Parameter'].isin(outcome_vars_to_exclude)]

        # 5. Crear encoding de features (después del filtrado)
        self.create_feature_encoding(patients_data)
        
        # 6. Crear secuencias
        X, times, masks, patient_ids = self.create_sequences(patients_data, max_seq_length)
        
        # 7. Normalizar features (solo las numéricas)
        X_normalized = self.normalize_features(X, masks, patients_data)
        
        # 8. Guardar en formato STraTS
        splits = self.save_strats_format(X_normalized, times, masks, labels, patient_ids, split_ratios)
        
        print("=== Pipeline completado exitosamente ===")
        print(f"Datos guardados en: {self.output_dir}")
        print(f"Shape final: {X_normalized.shape}")
        print(f"Features: {len(self.feature_to_idx)}")
        print(f"Secuencia max: {self.max_seq_length}")
        

        # ------------------------------------------------------------------
        # 9. (NEW) Guardar nombres de features en orden para interpretabilidad
        # ------------------------------------------------------------------
        features_ordered = [self.idx_to_feature[i] for i in range(len(self.idx_to_feature))]
        features_file = os.path.join(self.output_dir, 'features.json')
        with open(features_file, 'w', encoding='utf-8') as f:
            json.dump(features_ordered, f, indent=2, ensure_ascii=False)
        print(f"✓ features.json guardado en {features_file}")


        return splits, analysis

# Función principal para ejecutar el adaptador
def main():
    """
    Función principal para procesar los datos de tripletas y adaptarlos a STraTS
    """
    # Inicializar adaptador
    adapter = STraTSDataAdapter(
        tripletas_dir='tripletas',
        output_dir='strats_data'
    )
    
    # Procesar datos
    try:
        splits, analysis = adapter.process_complete_pipeline(
            max_seq_length=None,
            time_unit='weeks',
            split_ratios=(0.7, 0.15, 0.15)
        )
        
        print("\n=== Resumen del procesamiento ===")
        print(f"Total de pacientes: {len(splits['train']['patient_ids']) + len(splits['val']['patient_ids']) + len(splits['test']['patient_ids'])}")
        print(f"Train: {len(splits['train']['patient_ids'])}")
        print(f"Validation: {len(splits['val']['patient_ids'])}")
        print(f"Test: {len(splits['test']['patient_ids'])}")
        
    except Exception as e:
        print(f"Error durante el procesamiento: {e}")
        raise

if __name__ == "__main__":
    main()