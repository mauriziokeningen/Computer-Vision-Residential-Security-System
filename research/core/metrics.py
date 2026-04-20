import cv2
import numpy as np
import os
import random
from deepface import DeepFace
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# --- CONFIGURACIÓN ---
PATH_LFW = "lfw-deepfunneled\lfw-deepfunneled" # Tu carpeta LFW
FACIAL_MODEL = "ArcFace"
DETECTOR_BACKEND = "mtcnn"
MIN_IMAGES_REQUIRED = 15  # Solo probaremos con personas que tengan al menos 15 fotos
NUM_TEST_SUBJECTS = 5     # Probaremos con las 5 personas con más fotos del dataset

def get_all_images_recursive(root_folder, exclude_folder=None):
    """ Devuelve una lista de TODAS las imágenes en LFW excepto la carpeta del sujeto actual. """
    image_paths = []
    for root, dirs, files in os.walk(root_folder):
        if exclude_folder and exclude_folder in root:
            continue # Saltamos la carpeta del sujeto que estamos probando
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def generate_average_embedding(image_paths):
    embeddings = []
    for path in image_paths:
        try:
            res = DeepFace.represent(path, model_name=FACIAL_MODEL, detector_backend=DETECTOR_BACKEND, enforce_detection=True)
            embeddings.append(res[0]['embedding'])
        except:
            pass
    if not embeddings: return None
    return np.mean(embeddings, axis=0)

def get_similarity(img_path, ref_embedding):
    try:
        img = cv2.imread(img_path)
        if img is None: return 0.0
        res = DeepFace.represent(img, model_name=FACIAL_MODEL, detector_backend=DETECTOR_BACKEND, enforce_detection=True)
        live_emb = res[0]['embedding']
        return np.dot(live_emb, ref_embedding) / (np.linalg.norm(live_emb) * np.linalg.norm(ref_embedding))
    except:
        return 0.0

def evaluate_identity(name, folder_path, all_impostors):
    print(f"\n>>> Evaluando Robustez para: {name}")
    
    # 1. Obtener todas las fotos del sujeto
    all_photos = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    # 2. Dividir: 5 para referencia (Train), el resto para prueba (Test)
    train_imgs = all_photos[:5]
    test_positive_imgs = all_photos[5:]
    
    # 3. Crear Referencia
    ref_emb = generate_average_embedding(train_imgs)
    if ref_emb is None: return None

    y_true = []
    y_scores = []

    # 4. Probar Positivos (Deberían dar alto)
    # print(f"   Testing {len(test_positive_imgs)} fotos positivas...")
    for p in test_positive_imgs:
        s = get_similarity(p, ref_emb)
        y_true.append(1)
        y_scores.append(s)

    # 5. Probar Impostores (Deberían dar bajo)
    # Para no tardar años, tomamos 300 impostores aleatorios por cada sujeto
    impostor_sample = random.sample(all_impostors, 300)
    # print(f"   Testing {len(impostor_sample)} impostores aleatorios...")
    for p in impostor_sample:
        s = get_similarity(p, ref_emb)
        y_true.append(0)
        y_scores.append(s)

    # 6. Calcular F1 Máximo para este sujeto
    best_f1 = 0.0
    for thresh in np.arange(0.40, 0.90, 0.02):
        preds = [1 if s >= thresh else 0 for s in y_scores]
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1: best_f1 = f1
            
    print(f"   >>> Max F1-Score para {name}: {best_f1:.4f}")
    return best_f1

# --- MAIN ---
if __name__ == "__main__":
    # 1. Escanear carpetas para encontrar candidatos
    candidates = []
    print("Escaneando LFW para encontrar sujetos con suficientes fotos...")
    for person_name in os.listdir(PATH_LFW):
        person_folder = os.path.join(PATH_LFW, person_name)
        if os.path.isdir(person_folder):
            num_photos = len([f for f in os.listdir(person_folder) if f.endswith('.jpg')])
            if num_photos >= MIN_IMAGES_REQUIRED:
                candidates.append((person_name, num_photos, person_folder))
    
    # Ordenar por número de fotos (descendente) y tomar los Top N
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_subjects = candidates[:NUM_TEST_SUBJECTS]
    
    print(f"Sujetos seleccionados para la prueba: {[c[0] for c in top_subjects]}")

    # 2. Obtener lista maestra de todas las fotos (para usar de impostores)
    print("Indexando todas las imágenes para pool de impostores...")
    all_lfw_images = get_all_images_recursive(PATH_LFW)

    # 3. Loop de evaluación
    f1_results = []
    
    for name, count, folder in top_subjects:
        # Filtramos los impostores para que NO incluyan al sujeto actual
        current_impostors = [img for img in all_lfw_images if name not in img]
        
        score = evaluate_identity(name, folder, current_impostors)
        if score is not None:
            f1_results.append(score)

    # 4. Resultado Final de Robustez
    print("\n" + "="*40)
    print("RESULTADOS DE ROBUSTEZ DEL SISTEMA")
    print("="*40)
    print(f"Modelo: {FACIAL_MODEL}")
    print(f"Sujetos probados: {len(f1_results)}")
    print(f"F1-Scores individuales: {[f'{x:.2f}' for x in f1_results]}")
    print("-" * 40)
    print(f"PROMEDIO GLOBAL F1 (ROBUSTEZ): {np.mean(f1_results):.4f}")
    print("="*40)