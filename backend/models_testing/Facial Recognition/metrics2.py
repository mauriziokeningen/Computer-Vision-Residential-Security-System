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
# (Usamos la ruta que confirmaste que funciona)
PATH_LFW = r"C:\Users\Dassel\Desktop\TT\Modules\Facial Recognition\lfw-deepfunneled\lfw-deepfunneled"

# PARÁMETROS DEL MODELO
FACIAL_MODEL = "ArcFace"
DETECTOR_BACKEND = "mtcnn"

# CRITERIOS DE LA PRUEBA
MIN_IMAGES_REQUIRED = 20   # Solo usar sujetos con al menos 20 fotos (Datos sólidos)
IMPOSTORS_PER_SUBJECT = 100 # Número de impostores a probar contra cada sujeto

# ==========================================
#           FUNCIONES AUXILIARES
# ==========================================

def get_all_images_recursive(root_folder):
    """ Escanea recursivamente para obtener el pool de impostores """
    images = []
    print("Indexando banco de imágenes...")
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                images.append(os.path.join(root, file))
    return images

def get_embedding(img_path):
    """ Obtiene el vector facial usando DeepFace """
    try:
        res = DeepFace.represent(
            img_path=img_path, 
            model_name=FACIAL_MODEL, 
            detector_backend=DETECTOR_BACKEND, 
            enforce_detection=True
        )
        return res[0]['embedding']
    except:
        # Si no detecta cara, retorna None
        return None

def calculate_similarity(emb1, emb2):
    """ Calcula Similitud Coseno (0.0 a 1.0) """
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return np.dot(emb1, emb2) / (norm1 * norm2)

# ==========================================
#       LÓGICA DE REPORTE Y GRÁFICAS
# ==========================================

def plot_full_report(global_y_true, global_y_scores, subject_metrics):
    sns.set_style("whitegrid")
    
    print("\nCalculando curvas ROC y métricas avanzadas...")
    
    # 1. Calcular Curva ROC y AUC
    fpr, tpr, roc_thresholds = roc_curve(global_y_true, global_y_scores)
    roc_auc = auc(fpr, tpr)

    # 2. Calcular EER (Equal Error Rate) - El punto de oro biométrico
    try:
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer_thresh = interp1d(fpr, roc_thresholds)(eer)
    except:
        fnr = 1 - tpr
        eer_index = np.nanargmin(np.absolute((fnr - fpr)))
        eer = fpr[eer_index]
        eer_thresh = roc_thresholds[eer_index]

    # 3. Optimizar Umbral Operativo (Max F1)
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

    # 4. Calcular Métricas Finales
    tn, fp, fn, tp = confusion_matrix(global_y_true, global_preds).ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Acceptance Rate
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Rejection Rate
    accuracy = accuracy_score(global_y_true, global_preds)
    precision = precision_score(global_y_true, global_preds, zero_division=0)
    recall = recall_score(global_y_true, global_preds, zero_division=0)

    # --- DASHBOARD GRÁFICO (4 PANELES) ---
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Reporte Biométrico Integral ({FACIAL_MODEL}) - EER: {eer*100:.2f}%', fontsize=16)

    # A. Matriz de Confusión
    cm_display = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm_display, annot=True, fmt='d', cmap='Blues', ax=axs[0, 0],
                xticklabels=['Pred: Impostor', 'Pred: Genuino'],
                yticklabels=['Real: Impostor', 'Real: Genuino'])
    axs[0, 0].set_title(f'Matriz de Confusión (Umbral={final_thresh:.2f})')

    # B. Histograma de Distribución
    pos_scores = [s for s, t in zip(global_y_scores, global_y_true) if t == 1]
    neg_scores = [s for s, t in zip(global_y_scores, global_y_true) if t == 0]
    sns.histplot(pos_scores, color='green', label='Genuinos', kde=True, ax=axs[0, 1], binwidth=0.02, alpha=0.5)
    sns.histplot(neg_scores, color='red', label='Impostores', kde=True, ax=axs[0, 1], binwidth=0.02, alpha=0.5)
    axs[0, 1].axvline(final_thresh, color='black', linestyle='--', label=f'Umbral Op. ({final_thresh:.2f})')
    axs[0, 1].axvline(eer_thresh, color='orange', linestyle=':', label=f'Umbral EER ({eer_thresh:.2f})')
    axs[0, 1].set_title('Separabilidad de Clases (Histograma)')
    axs[0, 1].legend()

    # C. Curva ROC
    axs[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    axs[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[1, 0].scatter([eer], [1-eer], color='red', label='Punto EER', zorder=5)
    axs[1, 0].set_xlabel('False Positive Rate (FAR)')
    axs[1, 0].set_ylabel('True Positive Rate (Recall)')
    axs[1, 0].set_title('Curva ROC')
    axs[1, 0].legend(loc="lower right")

    # D. Robustez por Sujeto (Barras)
    subject_metrics.sort(key=lambda x: x['f1'], reverse=True)
    top_subjects = subject_metrics[:15] # Top 15
    names = [x['name'] for x in top_subjects]
    f1s = [x['f1'] for x in top_subjects]
    sns.barplot(x=f1s, y=names, palette='viridis', ax=axs[1, 1])
    axs[1, 1].set_xlim(0.8, 1.0)
    axs[1, 1].set_title('Consistencia por Sujeto (F1 Score)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- REPORTE TEXTUAL EN CONSOLA ---
    print("\n" + "="*60)
    print("       REPORTE DE RENDIMIENTO BIOMÉTRICO - ARC FACE")
    print("="*60)
    print(f"Modelo: {FACIAL_MODEL} | Backend: {DETECTOR_BACKEND}")
    print(f"Base de Datos: LFW | Sujetos Analizados: {len(subject_metrics)}")
    print(f"Total Muestras: {len(global_y_true)} (Genuinos: {tp+fn}, Impostores: {tn+fp})")
    print("-" * 60)
    print(f"1. EXACTITUD (Accuracy):          {accuracy*100:.2f}%")
    print(f"2. F1-SCORE (Balance):            {best_f1:.4f}")
    print(f"3. AUC-ROC (Potencia):            {roc_auc:.4f}")
    print(f"4. EER (Equal Error Rate):        {eer:.4f} ({eer*100:.2f}%) <-- MÉTRICA CLAVE")
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


# ==========================================
#           BLOQUE PRINCIPAL
# ==========================================
if __name__ == "__main__":
    
    # 1. BUSCAR SUJETOS VÁLIDOS
    print("--- 1. Escaneando LFW en busca de candidatos ---")
    if not os.path.exists(PATH_LFW):
        print(f"ERROR CRÍTICO: La ruta {PATH_LFW} no existe.")
        exit()

    valid_subjects = []
    for person_name in os.listdir(PATH_LFW):
        person_path = os.path.join(PATH_LFW, person_name)
        if os.path.isdir(person_path):
            # Solo contamos archivos jpg/png válidos
            images = [os.path.join(person_path, f) for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.png'))]
            if len(images) >= MIN_IMAGES_REQUIRED:
                valid_subjects.append({'name': person_name, 'images': images})
    
    print(f"Se encontraron {len(valid_subjects)} sujetos con al menos {MIN_IMAGES_REQUIRED} fotos.")
    
    # 2. INDEXAR POOL DE IMPOSTORES
    print("\n--- 2. Generando pool de impostores (Esto puede tardar un poco) ---")
    all_lfw_images = get_all_images_recursive(PATH_LFW)
    print(f"Total de imágenes en el banco: {len(all_lfw_images)}")

    # Variables globales para el reporte
    global_y_true = []
    global_y_scores = []
    subject_metrics_list = []

    # 3. EJECUCIÓN DEL BENCHMARK
    print("\n--- 3. Iniciando Benchmark Multi-Sujeto ---")
    
    # Usamos tqdm para barra de progreso general
    for subject in tqdm(valid_subjects, desc="Procesando Sujetos"):
        name = subject['name']
        photos = subject['images']
        
        # A. FASE DE ENROLLMENT (Referencia)
        # Usamos las primeras 5 fotos para crear la identidad
        ref_imgs = photos[:5]
        test_imgs = photos[5:] # El resto son para probar si el sistema lo reconoce
        
        # Calcular embeddings de referencia
        ref_embeddings = [get_embedding(p) for p in ref_imgs]
        # Filtrar los None (caras no detectadas)
        ref_embeddings = [e for e in ref_embeddings if e is not None]
        
        if not ref_embeddings: continue # Si no se pudo crear referencia, saltamos al siguiente
        
        # Promediar para crear el "Master Embedding"
        master_embedding = np.mean(ref_embeddings, axis=0)
        
        local_y_true = []
        local_y_scores = []

        # B. FASE DE TEST GENUINO (Debe dar 1)
        for p in test_imgs:
            emb = get_embedding(p)
            if emb is not None:
                score = calculate_similarity(emb, master_embedding)
                local_y_true.append(1)
                local_y_scores.append(score)
                
        # C. FASE DE TEST IMPOSTOR (Debe dar 0)
        # Seleccionar impostores que NO sean el sujeto actual
        possible_impostors = [x for x in all_lfw_images if name not in x]
        
        # Muestreo aleatorio para eficiencia
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

        # Agregar datos al acumulador global
        global_y_true.extend(local_y_true)
        global_y_scores.extend(local_y_scores)

        # Calcular métrica rápida para este sujeto
        if local_y_true:
            # Umbral temporal solo para monitoreo
            local_preds = [1 if s > 0.68 else 0 for s in local_y_scores]
            subject_f1 = f1_score(local_y_true, local_preds, zero_division=0)
            subject_metrics_list.append({'name': name, 'f1': subject_f1})

    # 4. GENERACIÓN FINAL
    if global_y_true:
        print("\nProcesamiento finalizado. Generando reporte...")
        plot_full_report(global_y_true, global_y_scores, subject_metrics_list)
    else:
        print("\nAdvertencia: No se generaron datos de prueba. Verifica las rutas y el contenido de las carpetas.")