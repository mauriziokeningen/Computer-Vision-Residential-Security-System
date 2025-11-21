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
try:
    from pose import (TinyPoseBiGRU, NTUDataset, make_splits, list_skeleton_files, 
                      A2TEXT, KEEP, action_id_from_filename)
    print("[INFO] Importación exitosa desde pose.py")
except ImportError as e:
    print(f"[ERROR] Fallo de importación: {e}")
    print("Asegúrate de que 'pose.py' esté en la misma carpeta que este script.")
    sys.exit()

# ==========================================
#   DICCIONARIO DE TRADUCCIÓN (EN -> ES)
# ==========================================
# Esto asegura que las gráficas salgan en español sin modificar pose.py
TRADUCCION_ACCIONES = {
    "Kicking": "Patear",
    "Hitting": "Golpear",
    "Punching/Slapping": "Puñetazo/Bofetada",
    "Pushing": "Empujar",
    "Staggering": "Tambalearse",
    "Falling": "Caerse",
    "Hand waving": "Saludar",
    "Pointing": "Señalar",
    # Variantes (si las hubiera en tu dataset)
    "Kicking (var)": "Patear (var)",
    "Hand waving (v2)": "Saludar (v2)",
    "Pointing (var)": "Señalar (var)"
}

def traducir_lista(lista_nombres):
    """Traduce una lista de nombres en inglés a español usando el diccionario."""
    return [TRADUCCION_ACCIONES.get(nombre, nombre) for nombre in lista_nombres]

# ==========================================
#      FUNCIÓN DE EVALUACIÓN
# ==========================================
def evaluate_preds(model, loader, device="cpu"):
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
#        LÓGICA DE GRAFICACIÓN (MODIFICADA)
# ==========================================
def plot_pose_metrics(y_true, y_pred, class_names_eng):
    sns.set_style("whitegrid")
    
    # 1. Traducir nombres al español para la gráfica
    class_names_es = traducir_lista(class_names_eng)
    
    # Calcular métricas globales
    acc = accuracy_score(y_true, y_pred)
    
    # Crear figura
    fig, axs = plt.subplots(2, 1, figsize=(12, 14))
    fig.suptitle(f'Validación de Análisis Corporal (TinyPoseBiGRU)\nExactitud Global: {acc:.2%}', fontsize=16)
    
    # --- 1. MATRIZ DE CONFUSIÓN (Heatmap Normalizado) ---
    cm = confusion_matrix(y_true, y_pred)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    # >>> AQUÍ SE CAMBIAN LOS COLORES Y TAMAÑO DE FUENTE <<<
    sns.heatmap(cm_norm, 
                annot=True,            # Mostrar números
                fmt='.2f',             # Formato de 2 decimales
                cmap='Reds',          # COLOR: 'Blues', 'Reds', 'Greens', 'Oranges', 'Purples'
                ax=axs[0],
                xticklabels=class_names_es, 
                yticklabels=class_names_es,
                annot_kws={"size": 10, "weight": "bold"}) # TAMAÑO y grosor de los números
    
    axs[0].set_title('Matriz de Confusión Normalizada (Recall por Acción)')
    axs[0].set_ylabel('Acción Real')
    axs[0].set_xlabel('Acción Predicha')
    axs[0].tick_params(axis='x', rotation=45)
    
    # --- 2. RENDIMIENTO POR CLASE (F1-Score) ---
    # Usamos los nombres en inglés para extraer del reporte, pero graficamos en español
    report = classification_report(y_true, y_pred, target_names=class_names_eng, output_dict=True)
    
    classes_plot_es = []
    f1_scores = []
    
    for k, v in report.items():
        if k in class_names_eng:
            # Traducir el nombre al español para la barra
            nombre_es = TRADUCCION_ACCIONES.get(k, k)
            classes_plot_es.append(nombre_es)
            f1_scores.append(v['f1-score'])
            
    # Ordenar por F1 para mejor visualización
    if len(f1_scores) > 0:
        sorted_indices = np.argsort(f1_scores)[::-1]
        classes_plot_es = [classes_plot_es[i] for i in sorted_indices]
        f1_scores = [f1_scores[i] for i in sorted_indices]
    
    # Colorear diferente las clases críticas (Buscando en español)
    critical_keywords = ['Patear', 'Golpe', 'Puñetazo', 'Empujón', 'Empujar']
    colors = ['#d62728' if any(x in c for x in critical_keywords) else '#1f77b4' for c in classes_plot_es]
    
    sns.barplot(x=f1_scores, y=classes_plot_es, palette=colors, ax=axs[1])
    axs[1].set_xlim(0, 1.0)
    axs[1].axvline(0.8, color='orange', linestyle='--', label='Objetivo TT1 (0.80)')
    axs[1].set_title('F1-Score por Tipo de Acción (Rojo = Crítico)')
    axs[1].set_xlabel('F1 Score')
    axs[1].legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_file = 'pose_validation_report.png'
    plt.savefig(output_file)
    print(f"\n--- [RESULTADOS] Gráfica guardada como: {output_file} ---")

# ==========================================
#        BLOQUE PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print("\n--- Iniciando Reporte de Métricas de Pose ---")
    
    # 1. GESTIÓN DE DATOS
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(current_dir, "NTU_dataset")
    skeletons_dir = os.path.join(dataset_root, "nturgb+d_skeletons")
    zip_filename = "nturgbd_skeletons_s001_to_s017.zip"
    zip_path = os.path.join(current_dir, zip_filename)

    DATA_DIR = None
    if os.path.exists(skeletons_dir) and len(os.listdir(skeletons_dir)) > 0:
        DATA_DIR = skeletons_dir
        print(f"[INFO] Datos detectados: {DATA_DIR}")
    else:
        print("[ADVERTENCIA] Buscando ZIP...")
        if os.path.exists(zip_path):
            print(f"[ARCHIVO] ZIP encontrado. Descomprimiendo...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    members = [m for m in z.namelist() if m.endswith(".skeleton")]
                    if not members: sys.exit("[ERROR] ZIP sin skeletons")
                    z.extractall(dataset_root, members=members)
                DATA_DIR = skeletons_dir
            except Exception as e:
                print(f"[ERROR] {e}")
                sys.exit()
        else:
            sys.exit("[ERROR CRÍTICO] No hay datos ni ZIP.")

    # 2. PREPARACIÓN
    all_files = list_skeleton_files(DATA_DIR)
    present_actions = sorted({action_id_from_filename(f) for f in all_files} & KEEP)
    ACTION2IDX  = {a:i for i,a in enumerate(present_actions)}
    _, test_files = make_splits(all_files, ACTION2IDX, test_size=0.2)
    
    print(f"Archivos de Test: {len(test_files)}")
    
    test_ds = NTUDataset(test_files, action2idx=ACTION2IDX, max_len=64, augment=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0) 
    
    # 3. MODELO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyPoseBiGRU(len(present_actions)).to(device)
    MODEL_PATH = os.path.join(current_dir, "best_pose_model.pth")
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("[INFO] Modelo cargado.")
    else:
        sys.exit("[ERROR] No se encontró 'best_pose_model.pth'. Entrena primero.")

    # 4. EJECUCIÓN
    all_y, all_p = evaluate_preds(model, test_loader, device)
    
    # Nombres en Inglés (Originales del modelo)
    ordered_names_eng = []
    for i in range(len(present_actions)):
        original_action_id = [k for k, v in ACTION2IDX.items() if v == i][0]
        ordered_names_eng.append(A2TEXT.get(original_action_id, f"Act {original_action_id}"))

    # 5. REPORTE GRÁFICO (Traduce internamente a Español)
    plot_pose_metrics(all_y, all_p, ordered_names_eng)
    
    # 6. REPORTE TEXTO (En Español)
    print("\n" + "="*60)
    print("      REPORTE DE CINEMÁTICA - TINYPOSE BiGRU")
    print("="*60)
    # Creamos lista de nombres en español para el reporte de texto
    ordered_names_es = traducir_lista(ordered_names_eng)
    print(classification_report(all_y, all_p, target_names=ordered_names_es, digits=4))
    print("="*60)