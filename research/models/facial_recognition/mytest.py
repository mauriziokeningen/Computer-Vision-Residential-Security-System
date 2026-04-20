import cv2
import numpy as np
from deepface import DeepFace
import os
import time
from collections import deque

# Intentar importar librería de NVIDIA para leer VRAM
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_GPU_MONITOR = True
except:
    HAS_GPU_MONITOR = False
    print("Advertencia: No se detectó GPU NVIDIA o falta 'nvidia-ml-py'. VRAM no disponible.")

# ==========================================
#               CONFIGURACIÓN
# ==========================================

PATH_REFERENCIAS = "mis_fotos" 
NOMBRE_USUARIO = "Mauricio"

# Backend recomendado para GPU
DETECTOR_BACKEND = "yolov8"  
FACIAL_MODEL = "ArcFace"
SIMILARITY_THRESHOLD = 0.40

# Configuración de Métricas
MOVING_AVG_WINDOW = 30 # Promediar los últimos 30 frames para estabilidad

# ==========================================
#           FUNCIONES AUXILIARES
# ==========================================

def get_gpu_memory():
    """ Obtiene el uso de VRAM en MB si hay GPU NVIDIA. """
    if not HAS_GPU_MONITOR:
        return 0
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) # GPU 0
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**2 # MB
    except:
        return 0

def draw_metrics(frame, avg_fps, avg_latency, vram):
    """ Dibuja el HUD de ingeniería. """
    
    # Estilo "Ingeniería"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    color_text = (0, 255, 0) # Verde consola
    thickness = 1
    bg_color = (0, 0, 0)
    
    # Métricas a mostrar
    metrics = [
        f"MODELO: {FACIAL_MODEL}",
        f"FPS (Avg): {avg_fps:.1f}",
        f"LATENCIA (Avg): {avg_latency:.1f} ms",
        f"VRAM GPU: {vram:.0f} MB" if vram > 0 else "VRAM: N/A (CPU)",
        "ESTADO: TIEMPO REAL"
    ]
    
    # Fondo semitransparente
    h, w, _ = frame.shape
    panel_h = 140
    panel_w = 280
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), bg_color, -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Escribir texto
    y_pos = 35
    for line in metrics:
        # Resaltar si la latencia es alta (>100ms es malo para tiempo real)
        if "LATENCIA" in line and avg_latency > 100:
            color_uso = (0, 0, 255) # Rojo si es lento
        else:
            color_uso = color_text
            
        cv2.putText(frame, line, (20, y_pos), font, font_scale, color_uso, thickness)
        y_pos += 25
        
    return frame

def generate_average_embedding(image_path_list):
    generated_embeddings = []
    print(f"Procesando {len(image_path_list)} imagenes para referencia...")
    for img_path in image_path_list:
        try:
            embedding_info = DeepFace.represent(
                img_path=img_path, model_name=FACIAL_MODEL, detector_backend=DETECTOR_BACKEND, enforce_detection=True
            )
            generated_embeddings.append(embedding_info[0]['embedding'])
        except: pass
    if not generated_embeddings: return None
    return np.mean(generated_embeddings, axis=0)

def load_my_reference():
    references = {}
    if not os.path.exists(PATH_REFERENCIAS): return {}
    image_paths = [os.path.join(PATH_REFERENCIAS, f) for f in os.listdir(PATH_REFERENCIAS) if f.lower().endswith(('.jpg','.png'))]
    if not image_paths: return {}
    avg_embedding = generate_average_embedding(image_paths)
    if avg_embedding is not None: references[NOMBRE_USUARIO] = avg_embedding
    return references

def find_best_match(live_embedding, references):
    best_name, best_similarity = "Desconocido", 0.0
    live_np = np.asarray(live_embedding)
    for name, ref_embedding in references.items():
        ref_np = np.asarray(ref_embedding)
        dot_product = np.dot(live_np, ref_np)
        norm_live = np.linalg.norm(live_np)
        norm_ref = np.linalg.norm(ref_np)
        similarity = dot_product / (norm_live * norm_ref) if norm_live > 0 and norm_ref > 0 else 0.0
        if similarity > best_similarity:
            best_similarity = similarity
            if similarity >= SIMILARITY_THRESHOLD: best_name = name
    return best_name, best_similarity

# ==========================================
#           BLOQUE PRINCIPAL
# ==========================================

if __name__ == "__main__":
    known_references = load_my_reference()
    if not known_references: exit()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): exit()

    # --- Buffers para promedio móvil (Estabilización) ---
    fps_buffer = deque(maxlen=MOVING_AVG_WINDOW)
    latency_buffer = deque(maxlen=MOVING_AVG_WINDOW)

    print("\n--- SISTEMA INICIADO ---")
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()

        try:
            # INFERENCIA
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=True
            )

            for face_obj in faces:
                if face_obj.get('confidence', 0) == 0: continue
                
                facial_area = face_obj['facial_area']
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                
                # Verificar que el recorte sea válido
                if w > 0 and h > 0 and y+h <= frame.shape[0] and x+w <= frame.shape[1]:
                    face_img = frame[y:y+h, x:x+w]
                    
                    embedding_results = DeepFace.represent(
                        img_path=face_img,
                        model_name=FACIAL_MODEL,
                        detector_backend="skip",
                        enforce_detection=False
                    )

                    if embedding_results:
                        current_embedding = embedding_results[0]['embedding']
                        name, score = find_best_match(current_embedding, known_references)

                        if name == NOMBRE_USUARIO:
                            color = (0, 255, 0)
                            label = f"{name} ({score:.2f})"
                        else:
                            color = (0, 0, 255)
                            label = f"Desconocido ({score:.2f})"

                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                        cv2.rectangle(display_frame, (x, y - 25), (x + w, y), color, cv2.FILLED)
                        cv2.putText(display_frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        except Exception:
            pass

        # CÁLCULO DE MÉTRICAS
        end_time = time.time()
        iteration_time = end_time - start_time
        
        # Evitar división por cero
        if iteration_time > 0:
            instant_fps = 1.0 / iteration_time
            instant_latency = iteration_time * 1000
        else:
            instant_fps = 0
            instant_latency = 0
            
        # Agregar a buffers para promediar
        fps_buffer.append(instant_fps)
        latency_buffer.append(instant_latency)
        
        # Calcular promedios
        avg_fps = sum(fps_buffer) / len(fps_buffer)
        avg_latency = sum(latency_buffer) / len(latency_buffer)
        vram_mb = get_gpu_memory()

        # DIBUJAR HUD
        display_frame = draw_metrics(display_frame, avg_fps, avg_latency, vram_mb)

        cv2.imshow('Sistema de Seguridad - Panel de Ingeniería', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()