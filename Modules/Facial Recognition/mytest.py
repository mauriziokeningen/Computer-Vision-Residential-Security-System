import cv2
import numpy as np
from deepface import DeepFace
import os

# ==========================================
#               CONFIGURACION
# ==========================================

# Carpeta donde pondras tus fotos de referencia
PATH_REFERENCIAS = "mis_fotos" 
NOMBRE_USUARIO = "Mauricio"

# Parametros del Sistema
# Nota: Para tiempo real, 'mtcnn' es muy preciso pero lento. 
# Si va lento, cambia a 'opencv', 'ssd' o 'yolov8'.
DETECTOR_BACKEND = "yolov8"  
FACIAL_MODEL = "ArcFace"
SIMILARITY_THRESHOLD = 0.39  # Ajustado segun tu calibracion previa (mas estricto)

# ===================================q=======
#           FUNCIONES AUXILIARES
# ==========================================

def generate_average_embedding(image_path_list):
    """ Genera un embedding promedio (Master Embedding) de tus fotos. """
    generated_embeddings = []
    print(f"Procesando {len(image_path_list)} imagenes para referencia...")
    
    for img_path in image_path_list:
        try:
            embedding_info = DeepFace.represent(
                img_path=img_path,
                model_name=FACIAL_MODEL,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True
            )
            generated_embeddings.append(embedding_info[0]['embedding'])
        except Exception as e:
            print(f"Saltando imagen {img_path}: No se detecto rostro claro.")
            
    if not generated_embeddings:
        return None
    # Promediar los vectores para crear una firma robusta
    return np.mean(generated_embeddings, axis=0)

def load_my_reference():
    """ Carga tus fotos de la carpeta y crea tu perfil. """
    references = {}
    
    if not os.path.exists(PATH_REFERENCIAS):
        print(f"ERROR: No existe la carpeta '{PATH_REFERENCIAS}'. Creala y pon tus fotos.")
        return {}

    # Buscar todas las imagenes jpg/png en la carpeta
    image_paths = [
        os.path.join(PATH_REFERENCIAS, f) 
        for f in os.listdir(PATH_REFERENCIAS) 
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]

    if not image_paths:
        print(f"ERROR: La carpeta '{PATH_REFERENCIAS}' esta vacia.")
        return {}

    avg_embedding = generate_average_embedding(image_paths)
    
    if avg_embedding is not None:
        references[NOMBRE_USUARIO] = avg_embedding
        print(f"Perfil de '{NOMBRE_USUARIO}' cargado exitosamente.")
    else:
        print("No se pudo crear el perfil. Verifica tus fotos.")
        
    return references

def find_best_match(live_embedding, references):
    """ Compara la cara de la webcam con la referencia. """
    best_name = "Desconocido"
    best_similarity = 0.0
    
    live_np = np.asarray(live_embedding)
    
    for name, ref_embedding in references.items():
        ref_np = np.asarray(ref_embedding)

        # Calculo de Similitud Coseno
        dot_product = np.dot(live_np, ref_np)
        norm_live = np.linalg.norm(live_np)
        norm_ref = np.linalg.norm(ref_np)
        
        if norm_live > 0 and norm_ref > 0:
            similarity = dot_product / (norm_live * norm_ref)
        else:
            similarity = 0.0

        if similarity > best_similarity:
            best_similarity = similarity
            if similarity >= SIMILARITY_THRESHOLD:
                best_name = name
                
    return best_name, best_similarity

# ==========================================
#           BLOQUE PRINCIPAL (WEBCAM)
# ==========================================

if __name__ == "__main__":
    # 1. Cargar Referencias
    known_references = load_my_reference()
    
    if not known_references:
        print("Saliendo porque no hay referencias...")
        exit()

    # 2. Iniciar Webcam (0 suele ser la camara default)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la webcam.")
        exit()

    print("\n--- SISTEMA INICIADO ---")
    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Copia del frame para dibujar
        display_frame = frame.copy()

        try:
            # 3. Detectar Rostros en el Frame actual
            # Usamos extract_faces primero para obtener coordenadas rapido
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=True
            )

            for face_obj in faces:
                # Si la confianza es 0 o no detecto nada real, saltar
                if face_obj.get('confidence', 0) == 0:
                    continue

                facial_area = face_obj['facial_area']
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                
                # Recortar cara para ArcFace
                face_img = frame[y:y+h, x:x+w]

                if face_img.size == 0: continue

                # 4. Extraer Embedding de la cara detectada
                embedding_results = DeepFace.represent(
                    img_path=face_img,
                    model_name=FACIAL_MODEL,
                    detector_backend="skip", # Ya detectamos, solo queremos embedding
                    enforce_detection=False
                )

                if embedding_results:
                    current_embedding = embedding_results[0]['embedding']
                    
                    # 5. Comparar
                    name, score = find_best_match(current_embedding, known_references)

                    # 6. Dibujar Resultados
                    # Verde si es Mauricio, Rojo si es Desconocido
                    if name == NOMBRE_USUARIO:
                        color = (0, 255, 0) # Verde BGR
                        label = f"{name} ({score:.2f})"
                    else:
                        color = (0, 0, 255) # Rojo BGR
                        label = f"Desconocido ({score:.2f})"

                    # Cuadro
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Etiqueta con fondo negro para leer mejor
                    cv2.rectangle(display_frame, (x, y - 25), (x + w, y), color, cv2.FILLED)
                    cv2.putText(display_frame, label, (x + 5, y - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        except Exception as e:
            # A veces DeepFace lanza error si no hay caras o movimiento brusco
            pass

        # 7. Mostrar Video
        cv2.imshow('Sistema de Seguridad - Reconocimiento Facial', display_frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()