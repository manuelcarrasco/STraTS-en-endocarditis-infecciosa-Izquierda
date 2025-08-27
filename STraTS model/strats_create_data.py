import pandas as pd
import os
import re
from typing import List
import shutil

def generate_tripletas_from_csv(csv_path: str = '1488_strats.csv', output_dir: str = 'tripletas'):
    """
    Genera archivos de tripletas a partir de un archivo CSV si no existen.
    Esta función ha sido movida desde el pipeline principal para centralizar la creación de datos.
    """
    print(f"Verificando la existencia de la carpeta de tripletas en '{output_dir}'...")

    if not os.path.exists(output_dir) or not os.listdir(output_dir):
        print(f"Generando tripletas desde '{csv_path}'...")

        if not os.path.exists(csv_path):
            print(f"Error: El archivo CSV de origen '{csv_path}' no fue encontrado.")
            return False

        # Cargar el archivo CSV
        df = pd.read_csv(csv_path, delimiter=';')

        # Crear directorio para guardar los archivos de texto
        os.makedirs(output_dir, exist_ok=True)

        # Iterar sobre cada fila del DataFrame
        for index, row in df.iterrows():
            num_protocolo = row['num_protocolo']
            tiempo_hasta_cirugia_val = row.get('tiempo_hasta_cirugia', 0)
            if pd.isna(tiempo_hasta_cirugia_val):
                tiempo_hasta_cirugia_val = 0

            t_muerte_alta_val = row.get('t_muerte_alta', 0)
            if pd.isna(t_muerte_alta_val):
                t_muerte_alta_val = 0

            # Lista para almacenar las tripletas del individuo actual
            tripletas_individuo = ["Time,Parameter,Value\n"]  # Encabezado

            # Iterar sobre cada columna para crear las tripletas
            for columna, valor in row.items():
                if pd.notna(valor):  # Solo procesar si el valor no es NaN
                    # Determinar el tiempo
                    if columna.startswith('ING'):
                        tiempo = 0
                    elif columna.startswith('S'):
                        # Extraer el número de la semana
                        if len(columna) > 1 and columna[1].isdigit():
                            tiempo = int(columna[1])  # Extraer el número de la semana
                        else:
                            tiempo = 0  # Asignar tiempo 0 si no es un número
                    elif columna == 'Cirugia':
                        tiempo = tiempo_hasta_cirugia_val
                    elif columna == 'Muerte_hospitalaria':
                        tiempo = t_muerte_alta_val  # Usar t_muerte_alta para Muerte_hospitalaria
                    else:
                        tiempo = 0  # Tiempo 0 para otras columnas

                    # Añadir la tripleta
                    tripletas_individuo.append(f"{tiempo},{columna},{valor}\n")

            # Guardar las tripletas en un archivo de texto
            with open(os.path.join(output_dir, f'{num_protocolo}.txt'), 'w') as archivo:
                archivo.writelines(tripletas_individuo)

        print(f"✓ Generados {len(df)} archivos de tripletas en '{output_dir}'")
    else:
        print("✓ Las tripletas ya existen, saltando la generación.")
    return True


class TripletasUnifier:
    """
    Clase para unificar nombres de variables en archivos de tripletas
    """
    
    def __init__(self, tripletas_dir: str = 'tripletas', backup_dir: str = 'tripletas_backup'):
        self.tripletas_dir = tripletas_dir
        self.backup_dir = backup_dir
        
        # Definir patrones de prefijos temporales a eliminar
        self.temporal_prefixes = [
            r'^ING\s+',      # ING Insuficiencia cardiaca -> Insuficiencia cardiaca
            r'^S\d+\s+',     # S1 Insuficiencia cardiaca -> Insuficiencia cardiaca
            r'^SEM\d+\s+',   # SEM1 Variable -> Variable
            r'^T\d+\s+',     # T1 Variable -> Variable
            r'^WEEK\d+\s+',  # WEEK1 Variable -> Variable
            r'^DIA\d+\s+',   # DIA1 Variable -> Variable
            r'^D\d+\s+',     # D1 Variable -> Variable
            r'^SEMANA\d+\s+', # SEMANA1 Variable -> Variable
            r'^MES\d+\s+',   # MES1 Variable -> Variable
            r'^M\d+\s+',     # M1 Variable -> Variable
        ]
    
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
    
    def create_backup(self):
        """
        Crear backup de la carpeta original
        """
        if os.path.exists(self.backup_dir):
            print(f"El backup {self.backup_dir} ya existe. Eliminando...")
            shutil.rmtree(self.backup_dir)
        
        print(f"Creando backup en {self.backup_dir}...")
        shutil.copytree(self.tripletas_dir, self.backup_dir)
        print("✓ Backup creado exitosamente")
    
    def analyze_unifications(self) -> dict:
        """
        Analizar qué unificaciones se realizarían sin modificar archivos
        """
        print("Analizando posibles unificaciones...")
        
        all_parameters = set()
        unifications = {}
        files_processed = 0
        
        for filename in os.listdir(self.tripletas_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.tripletas_dir, filename)
                
                try:
                    df = pd.read_csv(filepath)
                    if 'Parameter' in df.columns:
                        for param in df['Parameter'].dropna():
                            if isinstance(param, str):
                                all_parameters.add(param)
                                unified = self.unify_parameter_name(param)
                                if param != unified:
                                    unifications[param] = unified
                        files_processed += 1
                except Exception as e:
                    print(f"Error analizando {filename}: {e}")
        
        analysis = {
            'files_processed': files_processed,
            'total_unique_parameters': len(all_parameters),
            'unifications_needed': len(unifications),
            'unifications': unifications
        }
        
        return analysis
    
    def unify_all_files(self, create_backup: bool = True):
        """
        Unificar nombres de variables en todos los archivos de tripletas
        """
        if create_backup:
            self.create_backup()
        
        print("Iniciando unificación de archivos...")
        
        files_processed = 0
        files_modified = 0
        total_unifications = 0
        all_unifications = {}
        
        for filename in os.listdir(self.tripletas_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.tripletas_dir, filename)
                
                try:
                    df = pd.read_csv(filepath)
                    
                    if 'Parameter' in df.columns:
                        # Guardar parámetros originales para tracking
                        original_params = df['Parameter'].copy()
                        
                        # Unificar nombres
                        df['Parameter'] = df['Parameter'].apply(self.unify_parameter_name)
                        
                        # Contar cambios en este archivo
                        file_unifications = 0
                        for orig, unified in zip(original_params, df['Parameter']):
                            if pd.notna(orig) and pd.notna(unified) and orig != unified:
                                file_unifications += 1
                                if orig not in all_unifications:
                                    all_unifications[orig] = unified
                        
                        if file_unifications > 0:
                            # Guardar archivo modificado
                            df.to_csv(filepath, index=False)
                            files_modified += 1
                            total_unifications += file_unifications
                            print(f"  ✓ {filename}: {file_unifications} unificaciones")
                        
                        files_processed += 1
                    
                except Exception as e:
                    print(f"  ✗ Error procesando {filename}: {e}")
        
        print(f"\n=== Resumen de unificación ===")
        print(f"Archivos procesados: {files_processed}")
        print(f"Archivos modificados: {files_modified}")
        print(f"Total de unificaciones: {total_unifications}")
        
        if all_unifications:
            print(f"\n=== Unificaciones realizadas ({len(all_unifications)} únicas) ===")
            for orig, unified in sorted(all_unifications.items()):
                print(f"  '{orig}' -> '{unified}'")
        
        return {
            'files_processed': files_processed,
            'files_modified': files_modified,
            'total_unifications': total_unifications,
            'unique_unifications': all_unifications
        }
    
    def validate_unification(self):
        """
        Validar que la unificación se realizó correctamente
        """
        print("Validando unificación...")
        
        all_parameters = set()
        files_with_prefixes = []
        
        for filename in os.listdir(self.tripletas_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.tripletas_dir, filename)
                
                try:
                    df = pd.read_csv(filepath)
                    if 'Parameter' in df.columns:
                        file_has_prefixes = False
                        for param in df['Parameter'].dropna():
                            if isinstance(param, str):
                                all_parameters.add(param)
                                # Verificar si aún tiene prefijos
                                for prefix_pattern in self.temporal_prefixes:
                                    if re.match(prefix_pattern, param, flags=re.IGNORECASE):
                                        file_has_prefixes = True
                                        break
                        
                        if file_has_prefixes:
                            files_with_prefixes.append(filename)
                
                except Exception as e:
                    print(f"Error validando {filename}: {e}")
        
        print(f"Total de parámetros únicos después de unificación: {len(all_parameters)}")
        
        if files_with_prefixes:
            print(f"⚠️  Archivos que aún tienen prefijos temporales: {len(files_with_prefixes)}")
            for f in files_with_prefixes[:5]:  # Mostrar solo los primeros 5
                print(f"  - {f}")
            if len(files_with_prefixes) > 5:
                print(f"  ... y {len(files_with_prefixes) - 5} más")
        else:
            print("✓ Todos los archivos han sido unificados correctamente")
        
        return len(files_with_prefixes) == 0

def main():
    """
    Función principal para generar y unificar archivos de tripletas.
    """
    print("=== PIPELINE DE PREPARACIÓN DE DATOS (TRIPLETAS) ===\n")

    # --- PASO 1: Generar tripletas desde CSV si es necesario ---
    print("--- 1. Generación de tripletas ---")
    if not generate_tripletas_from_csv(csv_path='1488_strats.csv', output_dir='tripletas'):
        # Si la generación falla (p.ej. no encuentra el CSV), se detiene.
        return

    # --- PASO 2: Unificar nombres de variables ---
    print("\n--- 2. Unificación de variables ---")
    
    # Inicializar unificador
    unifier = TripletasUnifier(
        tripletas_dir='tripletas',
        backup_dir='tripletas_backup'
    )
    
    # Análisis previo
    print("\nAnalizando archivos actuales...")
    analysis = unifier.analyze_unifications()
    
    print(f"Archivos encontrados: {analysis['files_processed']}")
    print(f"Parámetros únicos: {analysis['total_unique_parameters']}")
    print(f"Unificaciones necesarias: {analysis['unifications_needed']}")
    
    if analysis['unifications_needed'] == 0:
        print("✓ No se necesitan unificaciones. Los archivos ya están limpios.")
        return
    
    print(f"\nEjemplos de unificaciones que se realizarán:")
    for i, (orig, unified) in enumerate(list(analysis['unifications'].items())[:5]):
        print(f"  '{orig}' -> '{unified}'")
    if len(analysis['unifications']) > 5:
        print(f"  ... y {len(analysis['unifications']) - 5} más")
    
    # Confirmación
    response = input(f"\n¿Proceder con la unificación? (s/N): ").strip().lower()
    if response not in ['s', 'si', 'sí', 'y', 'yes']:
        print("Operación cancelada.")
        return
    
    # Unificación
    print("\nRealizando unificación...")
    results = unifier.unify_all_files(create_backup=True)
    
    # Validación
    print("\nValidando resultados...")
    validation_ok = unifier.validate_unification()
    
    if validation_ok:
        print("\n✅ ¡Proceso de preparación de datos completado exitosamente!")
        print(f"Backup guardado en: tripletas_backup")
        print(f"Archivos finales generados en: tripletas")
    else:
        print("\n⚠️  La unificación puede no estar completa. Revisa los archivos manualmente.")

if __name__ == "__main__":
    main()
