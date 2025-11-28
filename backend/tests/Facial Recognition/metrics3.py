import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from deepface import DeepFace
from sklearn.metrics import (confusion_matrix, classification_report, f1_score, 
                             accuracy_score, precision_score, recall_score, 
                             roc_curve, auc)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm

# ==========================================
#               CONFIGURACIÓN
# ==========================================

# RUTA ABSOLUTA A TU DATASET LFW
PATH_LFW = r"C:\Users\Dassel\Desktop\TT\Modules\Facial Recognition\lfw-deepfunneled\lfw-deepfunneled"

# --- RUTAS DE GUARDADO DE DATOS ---
RESULTS_DIR = "facial_metrics"
SCORES_FILE = os.path.join(RESULTS_DIR, "global_scores.npy")
LABELS_FILE = os.path.join(RESULTS_DIR, "global_labels.npy")

# PARÁMETROS DEL MODELO
FACIAL_MODEL = "ArcFace"
DETECTOR_BACKEND = "mtcnn"

# CRITERIOS DE LA PRUEBA
MIN_IMAGES_REQUIRED = 20   
IMPOSTORS_PER_SUBJECT = 100 

# ==========================================
#          FUNCIONES AUXILIARES
# ==========================================

def get_all_images_recursive(root_folder):
    images = []
    print("Indexando banco de imágenes...")
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                images.append(os.path.join(root, file))
    return images

def get_embedding(img_path):
    try:
        res = DeepFace.represent(
            img_path=img_path, 
            model_name=FACIAL_MODEL, 
            detector_backend=DETECTOR_BACKEND, 
            enforce_detection=True
        )
        return res[0]['embedding']
    except:
        return None

def calculate_similarity(emb1, emb2):
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return np.dot(emb1, emb2) / (norm1 * norm2)

# ===================================================================
#             FUNCIÓN DE GRÁFICA EXTRA (FAR vs FRR)
# ===================================================================
def plot_far_frr_vs_threshold(fpr, tpr, roc_thresholds, eer_value):
    """
    Genera la curva de Tasas de Error en español.
    """
    far = fpr
    frr = 1 - tpr 
    
    # Encontrar el umbral correspondiente al EER para graficarlo
    try:
        eer_threshold = interp1d(fpr, roc_thresholds)(eer_value)
    except:
        eer_threshold = roc_thresholds[np.nanargmin(np.abs(fpr - (1 - tpr)))]

    plt.figure(figsize=(10, 6))
    
    # Graficar curvas
    plt.plot(roc_thresholds, frr, label='FRR (Tasa de Falsos Rechazos)', color='red', linewidth=2)
    plt.plot(roc_thresholds, far, label='FAR (Tasa de Falsos Aceptados)', color='green', linewidth=2)

    # Marcar puntos clave
    plt.scatter(eer_threshold, eer_value, color='blue', s=80, label=f'Punto EER ({eer_value:.4f})', zorder=5)
    plt.axvline(0.40, color='black', linestyle='--', label='Umbral Operativo (0.40)')

    plt.title('Curva de Balance de Error: FAR vs FRR')
    plt.xlabel('Umbral de Similitud Coseno')
    plt.ylabel('Tasa de Error (Probabilidad)')
    plt.ylim(0, 0.20) # Zoom al área de interés
    plt.xlim(0.0, 0.8)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_filename = 'FAR_FRR_Curve.png'
    plt.savefig(output_filename)
    print(f"\n[GRÁFICA EXTRA] Curva FAR/FRR guardada como: {output_filename}")
    plt.close() 


# ==========================================
#       LÓGICA DE REPORTE Y GRÁFICAS
# ==========================================

def plot_full_report(global_y_true, global_y_scores, subject_metrics):
    sns.set_style("whitegrid")
    
    print("\nCalculando curvas ROC y métricas avanzadas...")
    
    # 1. Calcular Curva ROC y AUC
    fpr, tpr, roc_thresholds = roc_curve(global_y_true, global_y_scores)
    roc_auc = auc(fpr, tpr)

    # 2. Calcular EER
    try:
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer_thresh = interp1d(fpr, roc_thresholds)(eer)
    except:
        fnr = 1 - tpr
        eer_index = np.nanargmin(np.absolute((fnr - fpr)))
        eer = fpr[eer_index]
        eer_thresh = roc_thresholds[eer_index]

    # 3. Generar Gráfica Extra (FAR/FRR) en Español
    plot_far_frr_vs_threshold(fpr, tpr, roc_thresholds, eer)

    # 4. Optimizar Umbral Operativo (Max F1)
    best_f1_thresh = 0
    best_f1 = 0
    thresholds_f1 = np.arange(0.0, 1.0, 0.01)
    for t in thresholds_f1:
        preds = [1 if s >= t else 0 for s in global_y_scores]
        f1 = f1_score(global_y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_thresh = t
            
    final_thresh = best_f1_thresh
    global_preds = [1 if s >= final_thresh else 0 for s in global_y_scores]

    # 5. Calcular Métricas Finales
    tn, fp, fn, tp = confusion_matrix(global_y_true, global_preds).ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0 
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0 
    accuracy = accuracy_score(global_y_true, global_preds)
    precision = precision_score(global_y_true, global_preds, zero_division=0)
    recall = recall_score(global_y_true, global_preds, zero_division=0)

    # --- DASHBOARD GRÁFICO (4 PANELES) EN ESPAÑOL ---
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Reporte Biométrico Integral ({FACIAL_MODEL}) - EER: {eer*100:.2f}%', fontsize=30)

    # A. Matriz de Confusión
    cm_display = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm_display, annot=True, fmt='d', cmap='Reds', ax=axs[0, 0],
                xticklabels=['Pred: Impostor', 'Pred: Genuino'],
                yticklabels=['Real: Impostor', 'Real: Genuino'])
    axs[0, 0].set_title(f'Matriz de Confusión (Umbral={final_thresh:.2f})')
    axs[0, 0].set_xlabel('Predicción del Sistema')
    axs[0, 0].set_ylabel('Identidad Real')

    # B. Histograma de Distribución
    pos_scores = [s for s, t in zip(global_y_scores, global_y_true) if t == 1]
    neg_scores = [s for s, t in zip(global_y_scores, global_y_true) if t == 0]
    sns.histplot(pos_scores, color='green', label='Genuinos (Residentes)', kde=True, ax=axs[0, 1], binwidth=0.02, alpha=0.5)
    sns.histplot(neg_scores, color='red', label='Impostores (Intrusos)', kde=True, ax=axs[0, 1], binwidth=0.02, alpha=0.5)
    axs[0, 1].axvline(final_thresh, color='black', linestyle='--', label=f'Umbral Op. ({final_thresh:.2f})')
    axs[0, 1].axvline(eer_thresh, color='orange', linestyle=':', label=f'Umbral EER ({eer_thresh:.2f})')
    axs[0, 1].set_title('Separabilidad de Clases (Histograma)')
    axs[0, 1].set_xlabel('Puntaje de Similitud Coseno')
    axs[0, 1].set_ylabel('Cantidad de Muestras')
    axs[0, 1].legend()

    # C. Curva ROC
    axs[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    axs[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[1, 0].scatter([eer], [1-eer], color='red', label='Punto EER', zorder=5)
    axs[1, 0].set_xlabel('Tasa de Falsos Positivos (FAR)')
    axs[1, 0].set_ylabel('Tasa de Verdaderos Positivos (Recall)')
    axs[1, 0].set_title('Curva ROC')
    axs[1, 0].legend(loc="lower right")

    # D. Robustez por Sujeto (Barras)
    # Ordenar y tomar top 15 para visualización limpia
    valid_metrics = [m for m in subject_metrics if m['f1'] > 0]
    valid_metrics.sort(key=lambda x: x['f1'], reverse=True)
    
    if len(valid_metrics) > 0:
        limit = min(15, len(valid_metrics))
        top_subjects = valid_metrics[:limit] 
        names = [x['name'] for x in top_subjects]
        f1s = [x['f1'] for x in top_subjects]
        sns.barplot(x=f1s, y=names, palette='viridis', ax=axs[1, 1])
        axs[1, 1].set_xlim(0.8, 1.0)
        axs[1, 1].set_title('Consistencia por Sujeto (F1 Score)')
        axs[1, 1].set_xlabel('Puntaje F1')
    else:
        axs[1, 1].text(0.5, 0.5, "Datos insuficientes para gráfica por sujeto", 
                       ha='center', va='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- REPORTE TEXTUAL EN CONSOLA (ESPAÑOL) ---
    print("\n" + "="*60)
    print("       REPORTE DE RENDIMIENTO BIOMÉTRICO - ARC FACE")
    print("="*60)
    print(f"Modelo: {FACIAL_MODEL} | Backend: {DETECTOR_BACKEND}")
    print(f"Total Muestras: {len(global_y_true)} (Genuinos: {tp+fn}, Impostores: {tn+fp})")
    print("-" * 60)
    print(f"1. EXACTITUD (Accuracy):          {accuracy*100:.2f}%")
    print(f"2. F1-SCORE (Balance):            {best_f1:.4f}")
    print(f"3. AUC-ROC (Potencia):            {roc_auc:.4f}")
    print(f"4. EER (Punto de Equilibrio):     {eer:.4f} ({eer*100:.2f}%) <-- MÉTRICA CLAVE")
    print("-" * 60)
    print("PUNTO DE OPERACIÓN ÓPTIMO (Max F1):")
    print(f"   - Umbral Configurado:          {final_thresh:.2f}")
    print(f"   - FAR (Falsos Aceptados):      {far*100:.4f}%  (Seguridad)")
    print(f"   - FRR (Falsos Rechazados):     {frr*100:.4f}%  (Conveniencia)")
    print("-" * 60)
    print("OTRAS MÉTRICAS:")
    print(f"   - Recall (Sensibilidad):       {recall:.4f}")
    print(f"   - Especificidad:               {specificity:.4f}")
    print(f"   - Precisión:                   {precision:.4f}")
    print("="*60)


# ===================================================================
#           BLOQUE PRINCIPAL: LÓGICA DE GUARDADO/CARGA
# ===================================================================

if __name__ == "__main__":
    
    # 1. Crear el directorio de guardado si no existe
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    # 2. INTENTO DE CARGA (Para desarrollo rápido)
    if os.path.exists(SCORES_FILE) and os.path.exists(LABELS_FILE):
        print(f"\n--- Datos cargados de archivos: {SCORES_FILE} ---")
        print("Generando gráficas en ESPAÑOL...")
        global_y_scores = np.load(SCORES_FILE)
        global_y_true = np.load(LABELS_FILE)
        
        # Simulamos lista de sujetos vacía porque no se guardó el detalle por sujeto
        subject_metrics_list = [] 
    
    # 3. EJECUCIÓN DEL BENCHMARK (Si los archivos NO existen)
    else:
        print("\n--- ARCHIVOS NO ENCONTRADOS. Iniciando Benchmark Completo ---")
        
        if not os.path.exists(PATH_LFW):
            print(f"ERROR CRÍTICO: La ruta {PATH_LFW} no existe.")
            exit()

        valid_subjects = []
        for person_name in os.listdir(PATH_LFW):
            person_path = os.path.join(PATH_LFW, person_name)
            if os.path.isdir(person_path):
                images = [os.path.join(person_path, f) for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.png'))]
                if len(images) >= MIN_IMAGES_REQUIRED:
                    valid_subjects.append({'name': person_name, 'images': images})
        
        print(f"Se encontraron {len(valid_subjects)} sujetos con al menos {MIN_IMAGES_REQUIRED} fotos.")
        
        print("\n--- Generando pool de impostores ---")
        all_lfw_images = get_all_images_recursive(PATH_LFW)

        global_y_true = []
        global_y_scores = []
        subject_metrics_list = []

        print("\n--- 3. Iniciando Benchmark Multi-Sujeto ---")
        
        for subject in tqdm(valid_subjects, desc="Procesando Sujetos"):
            name = subject['name']
            photos = subject['images']
            
            # Enrolamiento
            ref_imgs = photos[:5]
            test_imgs = photos[5:]
            
            ref_embeddings = [get_embedding(p) for p in ref_imgs]
            ref_embeddings = [e for e in ref_embeddings if e is not None]
            
            if not ref_embeddings: continue
            
            master_embedding = np.mean(ref_embeddings, axis=0)
            
            local_y_true = []
            local_y_scores = []

            # Test Genuino
            for p in test_imgs:
                emb = get_embedding(p)
                if emb is not None:
                    score = calculate_similarity(emb, master_embedding)
                    local_y_true.append(1)
                    local_y_scores.append(score)
                    
            # Test Impostor
            possible_impostors = [x for x in all_lfw_images if name not in x]
            
            if len(possible_impostors) > IMPOSTORS_PER_SUBJECT:
                chosen_impostors = random.sample(possible_impostors, IMPOSTORS_PER_SUBJECT)
            else:
                chosen_impostors = possible_impostors
            
            for p in chosen_impostors:
                emb = get_embedding(p)
                if emb is not None:
                    score = calculate_similarity(emb, master_embedding)
                    local_y_true.append(0)
                    local_y_scores.append(score)

            # Acumular
            global_y_true.extend(local_y_true)
            global_y_scores.extend(local_y_scores)

            # Métricas por sujeto
            if local_y_true:
                local_preds = [1 if s > 0.40 else 0 for s in local_y_scores] 
                subject_f1 = f1_score(local_y_true, local_preds, zero_division=0)
                subject_metrics_list.append({'name': name, 'f1': subject_f1})

        # 4. GUARDAR DATOS
        global_y_scores_np = np.array(global_y_scores)
        global_y_true_np = np.array(global_y_true)
        
        np.save(SCORES_FILE, global_y_scores_np)
        np.save(LABELS_FILE, global_y_true_np)
        
        print(f"\n--- Datos GUARDADOS para uso futuro en: {RESULTS_DIR} ---")
        
    # 5. LLAMADA FINAL AL REPORTE
    if len(global_y_true) > 0:
        plot_full_report(global_y_true, global_y_scores, subject_metrics_list)
    else:
        print("\nAdvertencia: No se generaron datos de prueba.")