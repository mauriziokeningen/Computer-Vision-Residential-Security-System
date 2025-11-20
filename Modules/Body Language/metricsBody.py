import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from torch.utils.data import DataLoader
import os
import zipfile
import sys

# ==========================================
#    IMPORTACIONES DEL MÓDULO ORIGINAL
# ==========================================
# Importamos las clases y configuraciones necesarias de tu archivo pose.py
try:
    from pose import (TinyPoseBiGRU, NTUDataset, make_splits, list_skeleton_files, 
                      A2TEXT, KEEP, action_id_from_filename)
    print("[INFO] Importación exitosa desde pose.py")
except ImportError as e:
    print(f"[ERROR] Fallo de importación: {e}")
    print("Asegúrate de que 'pose.py' esté en la misma carpeta que este script.")
    sys.exit()

# ==========================================
#      FUNCIÓN DE EVALUACIÓN
# ==========================================
def evaluate_preds(model, loader, device="cpu"):
    """
    Ejecuta inferencia sobre todo el dataloader y devuelve etiquetas reales y predichas.
    """
    model.eval()
    all_y, all_p = [], []
    print(f"Iniciando evaluación en {len(loader)} lotes...")
    with torch.no_grad():
        for batch_idx, (X, y, _) in enumerate(loader):
            X = X.to(device).float()
            y = y.to(device)
            logits = model(X)
            p = logits.argmax(1)
            all_y.extend(y.cpu().tolist())
            all_p.extend(p.cpu().tolist())
            
            if batch_idx % 10 == 0:
                print(f"Procesado lote {batch_idx}/{len(loader)}...", end='\r')
    print("\nEvaluación completada.")
    return np.array(all_y), np.array(all_p)

# ==========================================
#        LÓGICA DE GRAFICACIÓN
# ==========================================
def plot_pose_metrics(y_true, y_pred, class_names):
    sns.set_style("whitegrid")
    
    # Calcular métricas globales
    acc = accuracy_score(y_true, y_pred)
    
    # Crear figura
    fig, axs = plt.subplots(2, 1, figsize=(12, 14))
    fig.suptitle(f'Validación de Análisis Corporal (TinyPoseBiGRU)\nExactitud Global: {acc:.2%}', fontsize=16)
    
    # --- 1. MATRIZ DE CONFUSIÓN (Heatmap Normalizado) ---
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalizar para ver porcentajes (Recall) - Evitar división por cero
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Reds', ax=axs[0],
                xticklabels=class_names, yticklabels=class_names)
    axs[0].set_title('Matriz de Confusión Normalizada (Recall por Acción)')
    axs[0].set_ylabel('Acción Real')
    axs[0].set_xlabel('Acción Predicha')
    axs[0].tick_params(axis='x', rotation=45)
    
    # --- 2. RENDIMIENTO POR CLASE (F1-Score) ---
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    classes_plot = []
    f1_scores = []
    
    for k, v in report.items():
        if k in class_names:
            classes_plot.append(k)
            f1_scores.append(v['f1-score'])
            
    # Ordenar por F1 para mejor visualización
    if len(f1_scores) > 0:
        sorted_indices = np.argsort(f1_scores)[::-1]
        classes_plot = [classes_plot[i] for i in sorted_indices]
        f1_scores = [f1_scores[i] for i in sorted_indices]
    
    # Colorear diferente las clases críticas de seguridad
    # Buscamos palabras clave en inglés o español según tu diccionario A2TEXT
    critical_keywords = ['Kick', 'Punch', 'Hit', 'Push', 'Golpe', 'Patada', 'Empujón']
    colors = ['#d62728' if any(x in c for x in critical_keywords) else '#1f77b4' for c in classes_plot]
    
    sns.barplot(x=f1_scores, y=classes_plot, palette=colors, ax=axs[1])
    axs[1].set_xlim(0, 1.0)
    axs[1].axvline(0.8, color='orange', linestyle='--', label='Objetivo TT1 (0.80)')
    axs[1].set_title('F1-Score por Tipo de Acción (Rojo = Crítico)')
    axs[1].set_xlabel('F1 Score')
    axs[1].legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_file = 'pose_validation_report.png'
    plt.savefig(output_file)
    print(f"\n--- [RESULTADOS] Gráfica guardada como: {output_file} ---")
    # plt.show() # Descomenta si quieres que se abra la ventana

# ==========================================
#        BLOQUE PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print("\n--- Iniciando Reporte de Métricas de Pose ---")
    
    # ---------------------------------------------------------
    # 1. GESTIÓN DE DATOS (AUTO-REPARACIÓN)
    # ---------------------------------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(current_dir, "NTU_dataset")
    skeletons_dir = os.path.join(dataset_root, "nturgb+d_skeletons")
    zip_filename = "nturgbd_skeletons_s001_to_s017.zip"
    zip_path = os.path.join(current_dir, zip_filename)

    print(f"Directorio de trabajo: {current_dir}")

    # Lógica para encontrar DATA_DIR
    DATA_DIR = None
    
    # Caso A: Los datos ya están descomprimidos
    if os.path.exists(skeletons_dir) and len(os.listdir(skeletons_dir)) > 0:
        DATA_DIR = skeletons_dir
        print(f"[INFO] Datos detectados en carpeta existente: {DATA_DIR}")
    
    # Caso B: No están descomprimidos, buscar ZIP
    else:
        print("[ADVERTENCIA] No se encontraron datos descomprimidos. Buscando archivo ZIP...")
        
        if os.path.exists(zip_path):
            print(f"[ARCHIVO] ZIP encontrado: {zip_path}")
            print("[ESTADO] Descomprimiendo... (Esto puede tardar unos minutos, por favor espere)")
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    # Filtrar solo archivos .skeleton para no llenar el disco de basura si hay otros archivos
                    members = [m for m in z.namelist() if m.endswith(".skeleton")]
                    if not members:
                          print("[ERROR] El ZIP no contiene archivos .skeleton")
                          sys.exit()
                    
                    # Extraer
                    z.extractall(dataset_root, members=members)
                
                DATA_DIR = skeletons_dir
                print(f"[INFO] Extracción completada en: {DATA_DIR}")
            except Exception as e:
                print(f"[ERROR] Fallo al descomprimir: {e}")
                sys.exit()
        else:
            print("\n[ERROR CRÍTICO] FALTAN LOS DATOS")
            print(f"1. La carpeta de datos '{skeletons_dir}' está vacía o no existe.")
            print(f"2. Y no se encontró el archivo '{zip_filename}' en la carpeta actual.")
            print(f"   Ruta esperada del ZIP: {zip_path}")
            print(">>> SOLUCIÓN: Asegúrate de que el archivo .zip esté junto a metricsBody.py")
            sys.exit()

    # ---------------------------------------------------------
    # 2. PREPARACIÓN DE DATASETS
    # ---------------------------------------------------------
    print("Listando archivos...")
    all_files = list_skeleton_files(DATA_DIR)
    
    if len(all_files) == 0:
        print("[ERROR] Se encontró la carpeta pero no hay archivos .skeleton válidos dentro.")
        sys.exit()

    # Reconstruir los mapeos de índices (Misma lógica que entrenamiento)
    present_actions = sorted({action_id_from_filename(f) for f in all_files} & KEEP)
    ACTION2IDX  = {a:i for i,a in enumerate(present_actions)}
    
    # Obtener solo los archivos de TEST (usando la misma semilla 42 para consistencia)
    _, test_files = make_splits(all_files, ACTION2IDX, test_size=0.2)
    
    print(f"Archivos de Test encontrados: {len(test_files)}")
    
    # Crear el DataLoader de Test
    test_ds = NTUDataset(test_files, action2idx=ACTION2IDX, max_len=64, augment=False)
    # num_workers=0 para máxima compatibilidad en Windows
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0) 
    
    # ---------------------------------------------------------
    # 3. CARGA DEL MODELO
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    n_classes = len(present_actions)
    model = TinyPoseBiGRU(n_classes).to(device)
    
    MODEL_PATH = "best_pose_model.pth"
    
    # Buscar el modelo en la carpeta actual
    model_full_path = os.path.join(current_dir, MODEL_PATH)
    
    if os.path.exists(model_full_path):
        print(f"Cargando pesos desde {model_full_path}...")
        try:
            model.load_state_dict(torch.load(model_full_path, map_location=device))
            print("[INFO] Modelo cargado correctamente.")
        except Exception as e:
            print(f"[ERROR] Fallo al cargar los pesos del modelo: {e}")
            print("Es posible que la arquitectura del modelo haya cambiado desde el entrenamiento.")
            sys.exit()
    else:
        print(f"[ADVERTENCIA] No se encontró {MODEL_PATH}.")
        print("Asegúrate de haber corrido pose.py primero para entrenar y guardar el modelo.")
        sys.exit()

    # ---------------------------------------------------------
    # 4. EJECUCIÓN Y REPORTE
    # ---------------------------------------------------------
    all_y, all_p = evaluate_preds(model, test_loader, device)
    
    # Generar Nombres para las Gráficas
    ordered_names = []
    for i in range(n_classes):
        # Buscar qué acción original corresponde a este índice
        original_action_id = [k for k, v in ACTION2IDX.items() if v == i][0]
        # Usar el diccionario A2TEXT importado de pose.py
        ordered_names.append(A2TEXT.get(original_action_id, f"Act {original_action_id}"))

    # Generar Reporte Visual
    plot_pose_metrics(all_y, all_p, ordered_names)
    
    # Reporte de Texto
    print("\n" + "="*60)
    print("      REPORTE DE CINEMÁTICA - TINYPOSE BiGRU")
    print("="*60)
    # digits=4 para ver más precisión
    print(classification_report(all_y, all_p, target_names=ordered_names, digits=4))
    print("="*60)