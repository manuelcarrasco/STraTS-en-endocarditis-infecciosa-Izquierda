#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script modular para un análisis de supervivencia completo, con división en muestra
de entrenamiento y validación para evaluar el rendimiento del modelo.

Autor: Manuel
Fecha: 2025-08-14 (Versión v10.5 - Corregida la asignación de residuos)
"""

import os
import argparse
import logging
import re
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("survival_analysis_v11.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuración de Estilo para Gráficos ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

class SurvivalAnalysisPipeline:
    """
    Encapsula el pipeline completo para el análisis de supervivencia,
    incluyendo la división en entrenamiento y validación.
    """
    def __init__(self, data_path, time_col, event_col, output_dir='results'):
        self.data_path = data_path
        self.time_col = time_col
        self.event_col = event_col
        self.output_dir = output_dir
        
        self.df = None
        self.df_flattened = None
        self.df_train = None
        self.df_val = None
        
        self.univariate_results = None
        self.cox_model_stepwise = None
        self.cox_model_full = None
        self.final_cox_features = None
        self.flattened_feature_names = []
        self.cols_to_exclude = ['num_protocolo', 'tiempo_hasta_cirugia']

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Directorio de salida '{self.output_dir}' asegurado.")

    def preprocess_data(self):
        """Prepara los datos, recodificando y excluyendo variables."""
        if self.df is None: return
        logger.info("Iniciando pre-procesamiento de datos...")
        self.df['Cirugia_recodificada'] = self.df['Cirugia'].apply(
            lambda x: 'No' if x == 1 else ('Si' if pd.notna(x) and x > 1 else np.nan)
        )
        logger.info("Variable 'Cirugia' recodificada en 'Cirugia_recodificada'.")
        if 'Cirugia' in self.df.columns:
            self.df.drop(columns=['Cirugia'], inplace=True)
        self.df.drop(columns=[col for col in self.cols_to_exclude if col in self.df.columns], inplace=True)
        logger.info(f"Columnas excluidas del análisis: {self.cols_to_exclude}")

    def load_data(self):
        """1. Carga los datos y los pre-procesa."""
        logger.info(f"Cargando datos desde '{self.data_path}'...")
        try:
            self.df = pd.read_csv(self.data_path, delimiter=';')
            logger.info(f"Datos cargados. Shape: {self.df.shape}")
            self.preprocess_data()
        except FileNotFoundError:
            logger.error(f"Error: No se encontró el archivo en: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Ocurrió un error al cargar los datos: {e}")
            raise

    def recode_mortality_at_3_months(self, time_limit_days=90):
        """Recodifica la mortalidad a un tiempo límite de 90 días."""
        if self.df is None: return
        logger.info(f"Recodificando la mortalidad a un límite de {time_limit_days} días.")
        original_time_col, original_event_col = self.time_col, self.event_col
        new_time_col, new_event_col = 'tiempo_muerte_3meses', 'muerte_3meses'
        self.df[new_time_col] = np.minimum(self.df[original_time_col], time_limit_days)
        self.df[new_event_col] = self.df[original_event_col]
        condition = (self.df[original_time_col] > time_limit_days) & (self.df[original_event_col] == 1)
        self.df.loc[condition, new_event_col] = 0
        logger.info(f"Se han recodificado {condition.sum()} pacientes como 'censurados'.")
        logger.info(f"Nuevas columnas '{new_time_col}' y '{new_event_col}' creadas.")
        # Actualizar las columnas de tiempo y evento de la clase
        self.time_col = new_time_col
        self.event_col = new_event_col

    def split_data(self, test_size=0.15, random_state=42):
        """2. Divide los datos en conjuntos de entrenamiento y validación."""
        if self.df_flattened is None:
            logger.error("Los datos deben ser aplanados ('flattened') antes de la división.")
            return

        logger.info(f"Dividiendo los datos: {1-test_size:.0%} para entrenamiento, {test_size:.0%} para validación.")
        
        try:
            # Estratificar por la columna de evento para mantener la proporción
            self.df_train, self.df_val = train_test_split(
                self.df_flattened,
                test_size=test_size,
                random_state=random_state,
                stratify=self.df_flattened[self.event_col]
            )
            logger.info(f"Tamaño del conjunto de entrenamiento: {self.df_train.shape}")
            logger.info(f"Eventos en entrenamiento: {self.df_train[self.event_col].sum()} ({self.df_train[self.event_col].mean():.2%})")
            logger.info(f"Tamaño del conjunto de validación: {self.df_val.shape}")
            logger.info(f"Eventos en validación: {self.df_val[self.event_col].sum()} ({self.df_val[self.event_col].mean():.2%})")
        except Exception as e:
            logger.error(f"No se pudo dividir los datos. ¿Hay suficientes eventos para estratificar? Error: {e}")
            # Si la estratificación falla, se intenta una división simple
            self.df_train, self.df_val = train_test_split(self.df_flattened, test_size=test_size, random_state=random_state)
            logger.warning("Se realizó una división no estratificada como alternativa.")


    def descriptive_analysis(self, df, output_filename, cat_threshold=10):
        """3. Realiza un análisis descriptivo sobre un dataframe dado."""
        if df is None: return
        logger.info(f"Iniciando análisis descriptivo para '{output_filename}'...")
        descriptive_list = []
        cols_for_desc = [c for c in df.columns if c not in [self.time_col, self.event_col, 't_muerte_alta', 'Muerte_hospitalaria']]
        for col in cols_for_desc:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > cat_threshold:
                desc = df[col].describe(percentiles=[.25, .50, .75])
                descriptive_list.append({'Variable': col, 'Tipo': 'Continua', 'Estadística': 'count', 'Valor': desc['count']})
                for stat_name, value in desc.drop('count').items():
                    descriptive_list.append({'Variable': col, 'Tipo': 'Continua', 'Estadística': stat_name, 'Valor': f"{value:.2f}"})
            else:
                counts = df[col].value_counts()
                percentages = df[col].value_counts(normalize=True) * 100
                descriptive_list.append({'Variable': col, 'Tipo': 'Categórica', 'Estadística': 'count', 'Valor': df[col].count()})
                for category, count in counts.items():
                    descriptive_list.append({'Variable': col, 'Tipo': 'Categórica', 'Estadística': f"Freq: {category}", 'Valor': f"{count} ({percentages.get(category, 0):.2f}%)"})
        desc_stats_df = pd.DataFrame(descriptive_list)
        output_path = os.path.join(self.output_dir, output_filename)
        desc_stats_df.to_csv(output_path, index=False, sep=';')
        logger.info(f"Tabla descriptiva guardada en '{output_path}'.")

    def univariate_analysis(self, df, output_filename):
        """4. Realiza análisis univariante sobre un dataframe dado."""
        if df is None: return
        logger.info(f"Iniciando análisis univariante para '{output_filename}'...")
        results = []
        grupo_evento_1 = df[df[self.event_col] == 1]
        grupo_evento_0 = df[df[self.event_col] == 0]
        for col in df.columns:
            if col in [self.time_col, self.event_col, *self.cols_to_exclude, 't_muerte_alta', 'Muerte_hospitalaria']: continue
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() >= 10:
                data1, data0 = grupo_evento_1[col].dropna(), grupo_evento_0[col].dropna()
                if len(data1) < 3 or len(data0) < 3: continue
                sample = df[col].dropna()
                if len(sample) > 4999: sample = sample.sample(4999)
                _, p_norm = stats.shapiro(sample)
                test_name, stat, p_val = ("T-test", *stats.ttest_ind(data1, data0, equal_var=False)) if p_norm > 0.05 else ("Mann-Whitney U", *stats.mannwhitneyu(data1, data0))
                results.append({'Variable': col, 'Test': test_name, 'Statistic': stat, 'p-value': p_val})
            elif df[col].nunique() < 10:
                contingency_table = pd.crosstab(df[col], df[self.event_col])
                if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2: continue
                chi2, p_val, _, _ = stats.chi2_contingency(contingency_table)
                results.append({'Variable': col, 'Test': 'Chi-cuadrado', 'Statistic': chi2, 'p-value': p_val})
        self.univariate_results = pd.DataFrame(results).sort_values('p-value').reset_index(drop=True)
        output_path = os.path.join(self.output_dir, output_filename)
        self.univariate_results.to_csv(output_path, index=False, sep=';')
        logger.info(f"Tabla univariante guardada en '{output_path}'.")

    def flatten_temporal_features(self, temporal_prefixes):
        """5. Unifica variables temporales en una única columna por característica."""
        if self.df is None: return
        logger.info("Aplanando características temporales...")
        self.df_flattened = self.df.copy()
        all_temporal_cols = [col for prefix in temporal_prefixes for col in self.df.columns if col.startswith(prefix)]
        base_variable_names = set(re.sub(r'^\S+\s+', '', col).strip() for col in all_temporal_cols)
        for base_name in base_variable_names:
            if not base_name: continue
            related_cols = [col for col in all_temporal_cols if col.endswith(base_name)]
            if not related_cols: continue
            new_col_name = f"{base_name.replace(' ', '_')}_presente"
            self.df_flattened[new_col_name] = (self.df_flattened[related_cols] == 1).any(axis=1).astype(int)
            self.flattened_feature_names.append(new_col_name)
        self.df_flattened.drop(columns=all_temporal_cols, inplace=True, errors='ignore')
        logger.info("Aplanamiento de características completado.")

    def _add_at_risk_table_weekly(self, ax, df_var, var_name, time_col_weeks):
        """Añade tabla de riesgo semanalmente a un gráfico K-M."""
        max_weeks = int(df_var[time_col_weeks].max())
        time_points = np.arange(0, max_weeks + 2, 2)
        at_risk_data, group_labels = [], []
        groups = sorted(df_var[var_name].unique())
        for group in groups:
            group_df = df_var[df_var[var_name] == group]
            at_risk_counts = [(group_df[time_col_weeks] >= t).sum() for t in time_points]
            at_risk_data.append(at_risk_counts)
            group_labels.append(f"{group}")
        the_table = ax.table(cellText=at_risk_data, rowLabels=group_labels, colLabels=[f"{t}" for t in time_points],
                           cellLoc='center', rowLoc='center', loc='bottom', bbox=[0.0, -0.6, 1.0, 0.35])
        the_table.auto_set_font_size(False); the_table.set_fontsize(9)
        ax.text(-0.1, -0.45, "Nº en Riesgo", transform=ax.transAxes, fontsize=10, weight='bold', ha='left')

    def plot_kaplan_meier_curves(self, variables, df, output_subdir):
        """6. Genera curvas K-M para las variables especificadas en el dataframe dado."""
        if df is None: return
        logger.info(f"Generando curvas de Kaplan-Meier para '{output_subdir}'...")
        km_output_dir = os.path.join(self.output_dir, output_subdir)
        os.makedirs(km_output_dir, exist_ok=True)

        for var in variables:
            if var not in df.columns or df[var].nunique(dropna=True) < 2:
                logger.warning(f"Omitiendo K-M para '{var}' (no encontrada o < 2 categorías).")
                continue

            fig, ax = plt.subplots(figsize=(12, 9))
            df_var = df.dropna(subset=[self.time_col, self.event_col, var]).copy()
            time_col_weeks = self.time_col + '_semanas'
            df_var[time_col_weeks] = df_var[self.time_col] / 7

            colors = sns.color_palette("viridis", n_colors=df_var[var].nunique())
            for i, value in enumerate(sorted(df_var[var].unique())):
                mask = df_var[var] == value
                kmf = KaplanMeierFitter()
                kmf.fit(df_var[time_col_weeks][mask], df_var[self.event_col][mask], label=f'{var} = {value}')
                kmf.plot_survival_function(ax=ax, color=colors[i], ci_show=True, ci_alpha=0.15)

            try:
                results = multivariate_logrank_test(df_var[time_col_weeks], df_var[var], df_var[self.event_col])
                ax.set_title(f'Supervivencia a 3 Meses por {var}\nLog-Rank p-value: {results.p_value:.4f}', fontsize=16, pad=20)
            except Exception as e:
                logger.error(f"No se pudo calcular Log-Rank para '{var}': {e}")
                ax.set_title(f'Supervivencia a 3 Meses por {var}', fontsize=16, pad=20)

            ax.set_xlabel("Tiempo de seguimiento (Semanas)", fontsize=12)
            ax.set_ylabel("Probabilidad de Supervivencia", fontsize=12)
            ax.legend(title=var, fontsize=10); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            self._add_at_risk_table_weekly(ax, df_var, var, time_col_weeks)
            
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.4)
            output_path = os.path.join(km_output_dir, f'km_{var}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()
            logger.info(f"Curva K-M para '{var}' guardada en '{output_path}'")

    def _generate_cox_outputs(self, model, data, model_name, is_validation_set=False):
        """Genera y guarda todos los outputs para un modelo de Cox en un conjunto de datos."""
        logger.info(f"\n--- Evaluando modelo en: '{model_name}' ---")
        
        model_dir = os.path.join(self.output_dir, f'cox_{model_name}')
        os.makedirs(model_dir, exist_ok=True)

        # Las pruebas de bondad de ajuste se realizan solo en el set de entrenamiento.
        if not is_validation_set:
            logger.info("--- Realizando análisis del modelo ajustado (entrenamiento) ---")
            summary = model.summary
            logger.info(f"Resumen del modelo ajustado:\n{summary.to_string()}")
            summary.to_csv(os.path.join(model_dir, 'summary.csv'), sep=';')
            self.plot_forest_plot_with_details(model, os.path.join(model_dir, 'forest_plot.png'))
            
            # VIF, Schoenfeld, y Gronnesby-Borgan son para evaluar el ajuste del modelo.
            features = [col for col in data.columns if col not in [self.time_col, self.event_col]]
            X = data[features].astype(float)
            vif_data = pd.DataFrame({"feature": X.columns, "VIF": [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]})
            vif_data.to_csv(os.path.join(model_dir, 'vif_analysis.csv'), sep=';')
            
            self.plot_schoenfeld_residuals_individual(model, data, model_dir)
            self.perform_gronnesby_borgan_test(model, data, model_dir)

        # Las métricas de rendimiento predictivo se calculan para ambos sets.
        logger.info("--- Evaluando rendimiento predictivo ---")
        c_index = model.score(data, scoring_method="concordance_index")
        logger.info(f"C-Index (Índice de Concordancia): {c_index:.4f}")

        median_time = data[self.time_col].median()
        if median_time > 0:
            self.plot_calibration_curve(model, data, median_time, model_dir)
        self.plot_time_dependent_roc(model, data, model_dir)


    def plot_forest_plot_with_details(self, model, output_path):
        """Genera un Forest Plot con HR, IC y p-valor."""
        summary = model.summary
        if summary.empty:
            logger.warning("El resumen del modelo está vacío. No se puede generar el Forest Plot.")
            return
            
        fig, ax = plt.subplots(figsize=(12, len(summary) * 0.6 + 2))
        y_pos = np.arange(len(summary))
        
        lower_error = summary['exp(coef)'] - summary['exp(coef) lower 95%']
        upper_error = summary['exp(coef) upper 95%'] - summary['exp(coef)']
        asymmetric_error = np.array([lower_error, upper_error])

        ax.errorbar(x=summary['exp(coef)'], y=y_pos, xerr=asymmetric_error,
                    fmt='o', capsize=5, color='black', ecolor='gray', elinewidth=1)

        ax.axvline(x=1, linestyle='--', color='red', linewidth=1)
        ax.set_yticks(y_pos); ax.set_yticklabels(summary.index)
        ax.set_xlabel('Hazard Ratio (HR)'); ax.set_xscale('log')
        ax.set_title('Forest Plot de Hazard Ratios', pad=20)

        for i, row in summary.iterrows():
            hr = f"{row['exp(coef)']:.2f}"
            ci = f"({row['exp(coef) lower 95%']:.2f} - {row['exp(coef) upper 95%']:.2f})"
            p_val = f"p={row['p']:.3f}" if row['p'] >= 0.001 else "p<0.001"
            ax.text(ax.get_xlim()[1] * 1.1, y_pos[summary.index.get_loc(i)], f"{hr} {ci}, {p_val}", va='center')
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(output_path, dpi=300); plt.close()
        logger.info(f"Forest plot detallado guardado en '{output_path}'")

    def plot_schoenfeld_residuals_individual(self, model, data, model_dir):
        """Genera gráficos de residuos de Schoenfeld para cada variable."""
        schoenfeld_dir = os.path.join(model_dir, 'schoenfeld_residuals')
        os.makedirs(schoenfeld_dir, exist_ok=True)
        logger.info("Realizando test de Riesgos Proporcionales (Schoenfeld)...")
        
        try:
            schoenfeld_results = model.check_assumptions(data, p_value_threshold=0.05, show_plots=False)
            logger.info("Resultados del test de Schoenfeld global:")
            for var, p_val, _ in schoenfeld_results:
                logger.info(f"  - Variable: {var}, p-valor: {p_val:.4f}{' (POTENCIAL VIOLACIÓN)' if p_val < 0.05 else ''}")

            residuals = model.compute_residuals(data, kind='schoenfeld')
            for var in residuals.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.regplot(x=residuals.index, y=residuals[var], lowess=True, 
                            scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
                ax.set_title(f'Residuos de Schoenfeld para: {var}')
                ax.set_xlabel('Tiempo'); ax.set_ylabel(f'Residuos de {var}')
                plt.tight_layout()
                plt.savefig(os.path.join(schoenfeld_dir, f'{var}.png'), dpi=300); plt.close()
            logger.info(f"Gráficos de residuos de Schoenfeld individuales guardados en '{schoenfeld_dir}'")
        except Exception as e:
            logger.error(f"No se pudieron generar los gráficos de Schoenfeld: {e}")

    def run_cox_analysis(self, p_threshold=0.05):
        """7. Realiza regresión de Cox (completo y por pasos) y valida los modelos."""
        if self.df_train is None or self.df_val is None:
            logger.error("Los datos de entrenamiento/validación no están disponibles.")
            return
            
        logger.info("Iniciando análisis de regresión de Cox...")
        
        features_to_consider = [col for col in self.df_train.columns if col not in [self.time_col, self.event_col, 't_muerte_alta', 'Muerte_hospitalaria', *self.cols_to_exclude]]
        
        # Preparar datos de entrenamiento
        data_train_dummies = pd.get_dummies(self.df_train[features_to_consider + [self.time_col, self.event_col]], 
                                            drop_first=True, dummy_na=False).dropna()
        
        # Preparar datos de validación, asegurando que las columnas coincidan
        data_val_dummies = pd.get_dummies(self.df_val[features_to_consider + [self.time_col, self.event_col]], 
                                          drop_first=True, dummy_na=False).dropna()
        data_val_dummies = data_val_dummies.reindex(columns=data_train_dummies.columns, fill_value=0)

        if data_train_dummies.empty or data_train_dummies[self.event_col].sum() == 0:
            logger.error("No hay datos o eventos en el set de entrenamiento para ajustar el modelo de Cox. Abortando.")
            return

        all_features = [col for col in data_train_dummies.columns if col not in [self.time_col, self.event_col]]

        # --- Modelo Completo ---
        logger.info("\n=== AJUSTANDO MODELO COMPLETO (TODAS LAS VARIABLES) ===")
        self.cox_model_full = CoxPHFitter()
        try:
            self.cox_model_full.fit(data_train_dummies, duration_col=self.time_col, event_col=self.event_col)
            # Evaluar en entrenamiento
            self._generate_cox_outputs(self.cox_model_full, data_train_dummies, "full_train", is_validation_set=False)
            # Evaluar en validación
            self._generate_cox_outputs(self.cox_model_full, data_val_dummies, "full_validation", is_validation_set=True)
        except Exception as e:
            logger.error(f"Error al ajustar o validar el modelo completo: {e}")

        # --- Modelo por Pasos (Stepwise) ---
        logger.info("\n=== AJUSTANDO MODELO POR PASOS (SELECCIÓN HACIA ATRÁS) ===")
        remaining_features = all_features.copy()
        while len(remaining_features) > 0:
            cph = CoxPHFitter()
            try:
                current_data = data_train_dummies[remaining_features + [self.time_col, self.event_col]]
                cph.fit(current_data, duration_col=self.time_col, event_col=self.event_col)
                max_p = cph.summary['p'].max()
                if max_p > p_threshold:
                    feature_to_remove = cph.summary['p'].idxmax()
                    remaining_features.remove(feature_to_remove)
                    logger.info(f"Eliminando variable '{feature_to_remove}' con p-valor de {max_p:.4f}")
                else: break
            except Exception as e:
                logger.error(f"Error durante selección de variables: {e}"); break
        
        if not remaining_features:
            logger.error("No se seleccionaron características para el modelo por pasos."); return
        
        self.final_cox_features = remaining_features
        logger.info(f"Variables finales seleccionadas para el modelo por pasos: {self.final_cox_features}")
        
        self.cox_model_stepwise = CoxPHFitter()
        final_data_train = data_train_dummies[self.final_cox_features + [self.time_col, self.event_col]].copy()
        final_data_val = data_val_dummies[self.final_cox_features + [self.time_col, self.event_col]].copy()
        
        self.cox_model_stepwise.fit(final_data_train, duration_col=self.time_col, event_col=self.event_col)
        # Evaluar en entrenamiento
        self._generate_cox_outputs(self.cox_model_stepwise, final_data_train, "stepwise_train", is_validation_set=False)
        # Evaluar en validación
        self._generate_cox_outputs(self.cox_model_stepwise, final_data_val, "stepwise_validation", is_validation_set=True)
        
        # --- Modelo Manual ---
        logger.info("\n=== AJUSTANDO MODELO MANUAL ===")
        # Se definen las variables para el modelo manual
        manual_features = ['Cirugia_recodificada', 'Edad', 'Insuficiencia_cardiaca_presente', 'Insuficiencia_renal_presente']
        # Se define la variable de estrato
        strata_var = 'Shock_septico_presente'
        
        # Asegurarse de que las variables existen en el dataframe aplanado
        all_manual_features = manual_features + [strata_var]
        missing_vars = [v for v in all_manual_features if v not in self.df_train.columns]
        if missing_vars:
            logger.error(f"Error: Las siguientes variables manuales no se encontraron en los datos de entrenamiento: {missing_vars}. No se puede ajustar el modelo manual.")
            return
            
        # Preparar los datos de entrenamiento para el modelo manual
        data_manual_train = self.df_train[all_manual_features + [self.time_col, self.event_col]].copy()
        data_manual_train = pd.get_dummies(data_manual_train, drop_first=True, dummy_na=False).dropna()
        
        # Preparar los datos de validación para el modelo manual
        data_manual_val = self.df_val[all_manual_features + [self.time_col, self.event_col]].copy()
        data_manual_val = pd.get_dummies(data_manual_val, drop_first=True, dummy_na=False).dropna()
        data_manual_val = data_manual_val.reindex(columns=data_manual_train.columns, fill_value=0)
        
        manual_model = CoxPHFitter()
        try:
            # Ajustar el modelo de Cox con estrato
            manual_model.fit(data_manual_train, duration_col=self.time_col, event_col=self.event_col, strata=[strata_var])
            # Evaluar en entrenamiento
            self._generate_cox_outputs(manual_model, data_manual_train, "manual_train", is_validation_set=False)
            # Evaluar en validación
            self._generate_cox_outputs(manual_model, data_manual_val, "manual_validation", is_validation_set=True)
        except Exception as e:
            logger.error(f"Error al ajustar o validar el modelo manual: {e}")


    def perform_gronnesby_borgan_test(self, model, data, output_dir, n_groups=10):
        """
        Realiza el test de bondad de ajuste de Gronnesby-Borgan.
        Este test solo debe ejecutarse en los datos de entrenamiento.
        """
        logger.info("Realizando test de bondad de ajuste de Gronnesby-Borgan...")
        try:
            df = data.copy()
            features = list(model.params_.index)
            
            # 1. Predecir el riesgo usando solo las features del modelo.
            df['risk_score'] = model.predict_partial_hazard(df[features])

            # 2. Calcular residuos. La función espera los mismos datos con los que se ajustó.
            martingale_residuals = model.compute_residuals(data, kind='martingale')
            
            # 3. Asignar la serie de residuos.
            # El resultado debería ser una Serie, pero los errores indican que puede ser un DataFrame.
            # Nos aseguramos de tener una Serie antes de la asignación para evitar errores.
            if isinstance(martingale_residuals, pd.DataFrame):
                logger.debug("compute_residuals devolvió un DataFrame, seleccionando la primera columna.")
                residuals_series = martingale_residuals.iloc[:, 0]
            else:
                residuals_series = martingale_residuals
            
            # Asignar la Serie a la nueva columna. Pandas se encarga de alinear por índice.
            df['martingale_resid'] = residuals_series
            
            # 4. Continuar con el test de G-B
            df['risk_group'] = pd.qcut(df['risk_score'], q=n_groups, labels=False, duplicates='drop')
            actual_n_groups = df['risk_group'].nunique()
            
            if actual_n_groups < 2:
                logger.warning("No se pudieron crear suficientes grupos de riesgo para el test de Gronnesby-Borgan.")
                return

            group_resid_sum = df.groupby('risk_group')['martingale_resid'].sum()
            test_statistic = np.sum(group_resid_sum ** 2)
            
            # Grados de libertad es g-2
            dof = actual_n_groups - 2
            if dof < 1:
                logger.warning(f"Grados de libertad insuficientes ({dof}) para el test de Gronnesby-Borgan.")
                return

            p_value = stats.chi2.sf(test_statistic, df=dof)
            logger.info(f"  Resultado Gronnesby-Borgan -> Chi2: {test_statistic:.4f}, p-valor: {p_value:.4f} (grupos={actual_n_groups}, gl={dof})")

        except Exception as e:
            logger.error(f"Error en test Gronnesby-Borgan: {e}")
            logger.error(traceback.format_exc())


    def plot_calibration_curve(self, model, data, time_horizon, output_dir, n_bins=10):
        logger.info(f"Generando gráfico de calibración (Horizonte: {time_horizon:.0f} días).")
        try:
            # Extraer solo las covariables del modelo para la predicción.
            features = list(model.params_.index)
            X_for_prediction = data[features]
            predicted_survival = model.predict_survival_function(X_for_prediction, times=[time_horizon]).iloc[0]

            cal_df = pd.DataFrame({'predicted': predicted_survival, self.time_col: data[self.time_col], self.event_col: data[self.event_col]})
            cal_df['bin'] = pd.qcut(cal_df['predicted'], q=n_bins, labels=False, duplicates='drop')
            mean_predicted = cal_df.groupby('bin')['predicted'].mean()
            observed_km = [KaplanMeierFitter().fit(g[self.time_col], g[self.event_col]).predict(time_horizon) for i, g in cal_df.groupby('bin')]
            plt.figure(figsize=(8, 8))
            plt.plot([0, 1], [0, 1], 'k--', label='Calibración Perfecta')
            plt.plot(mean_predicted, observed_km, 'o-', label='Modelo de Cox')
            plt.xlabel('Prob. Supervivencia Predicha'); plt.ylabel('Prob. Supervivencia Observada (K-M)')
            plt.title(f'Gráfico de Calibración (Horizonte: {time_horizon:.0f} días)')
            plt.legend(); plt.grid(True)
            output_path = os.path.join(output_dir, f'calibration_plot_{time_horizon:.0f}d.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()
            logger.info(f"Gráfico de calibración guardado en '{output_path}'")
        except Exception as e:
            logger.error(f"No se pudo generar el gráfico de calibración: {e}")

    def plot_time_dependent_roc(self, model, data, output_dir):
        logger.info("Generando curvas ROC tiempo-dependientes...")
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            logger.error("Scikit-learn no instalado. No se puede generar ROC tiempo-dependiente."); return
        
        df = data.copy()

        # Extraer solo las covariables del modelo para la predicción.
        features = list(model.params_.index)
        X_for_prediction = data[features]
        df['marker'] = model.predict_partial_hazard(X_for_prediction)

        event_times = df[df[self.event_col] == 1][self.time_col]
        if len(event_times) < 2: return
        time_points = np.linspace(event_times.min(), event_times.max(), 15)
        aucs = []
        for t in time_points:
            y_true = (df[self.time_col] <= t) & (df[self.event_col] == 1)
            at_risk = df[self.time_col] > t
            df_eval = df[y_true | at_risk]
            y_true_eval = (df_eval[self.time_col] <= t) & (df_eval[self.event_col] == 1)
            if len(np.unique(y_true_eval)) < 2: aucs.append(np.nan)
            else: aucs.append(roc_auc_score(y_true_eval, df_eval['marker']))
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, aucs, marker='o', linestyle='-')
        plt.title('Curva ROC Tiempo-Dependiente (AUC(t))'); plt.xlabel('Tiempo (días)'); plt.ylabel('AUC')
        plt.ylim(0.4, 1.0); plt.grid(True)
        output_path = os.path.join(output_dir, 'time_dependent_roc.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()
        logger.info(f"Gráfico ROC tiempo-dependiente guardado en '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Pipeline Modular de Análisis de Supervivencia con Validación.")
    parser.add_argument('--data_path', type=str, default='1488_strats.csv', help="Ruta al archivo de datos CSV.")
    parser.add_argument('--time_col', type=str, default='t_muerte_alta', help="Nombre de la columna de tiempo original.")
    parser.add_argument('--event_col', type=str, default='Muerte_hospitalaria', help="Nombre de la columna de evento original.")
    parser.add_argument('--output_dir', type=str, default='resultados_analisis_3_meses_v11', help="Directorio para guardar los resultados.")
    args = parser.parse_args()

    pipeline = SurvivalAnalysisPipeline(
        data_path=args.data_path,
        time_col=args.time_col,
        event_col=args.event_col,
        output_dir=args.output_dir
    )
    
    # 1. Cargar y pre-procesar datos iniciales
    pipeline.load_data()
    pipeline.recode_mortality_at_3_months()
    
    # 2. Aplanar características temporales (antes de dividir)
    temporal_prefixes = ['ING ', 'S1 ', 'S2 ', 'S3 ', 'S4 ', 'S6 '] 
    pipeline.flatten_temporal_features(temporal_prefixes=temporal_prefixes)
    
    # 3. Dividir en entrenamiento y validación
    pipeline.split_data()
    
    # 4. Realizar análisis descriptivo y univariante en la muestra de entrenamiento
    pipeline.descriptive_analysis(df=pipeline.df_train, output_filename='tabla_descriptiva_entrenamiento.csv')
    pipeline.univariate_analysis(df=pipeline.df_train, output_filename='tabla_univariante_entrenamiento.csv')
    
    # 5. Graficar curvas de Kaplan-Meier en la muestra de entrenamiento
    if pipeline.df_train is not None:
        potential_other_vars = ['Sexo', 'Cirugia_recodificada', 'micro5']
        existing_other_vars = [v for v in potential_other_vars if v in pipeline.df_train.columns]
        km_vars_to_plot = list(dict.fromkeys(pipeline.flattened_feature_names + existing_other_vars))
        pipeline.plot_kaplan_meier_curves(variables=km_vars_to_plot, df=pipeline.df_train, output_subdir='kaplan_meier_entrenamiento')
    
    # 6. Ajustar modelos de Cox con entrenamiento y evaluar en validación
    pipeline.run_cox_analysis()

    logger.info("--- El pipeline de análisis de supervivencia ha finalizado ---")

if __name__ == '__main__':
    main()
