import cv2
import numpy as np
#import psycopg2
from deepface import DeepFace
import os

# --- CONFIGURATION ---
IMAGE_TO_TEST = "Modules/Facial Recognition/photos/test/Abdullah_Gul/Abdullah_Gul_0002.jpg" 

# --- System Constants ---
SIMILARITY_THRESHOLD = 0.68  # Threshold for ArcFace.
FACIAL_MODEL = "ArcFace"
DETECTOR_BACKEND = "mtcnn"

def generate_average_embedding(image_path_list):
    """
    Helper function to generate a robust, averaged embedding from multiple images.
    """
    generated_embeddings = []
    for img_path in image_path_list:
        try:
            embedding_info = DeepFace.represent(
                img_path=img_path,
                model_name=FACIAL_MODEL,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True
            )
            generated_embeddings.append(embedding_info[0]['embedding'])
        except Exception:
            print(f"Warning: Could not process reference image {img_path}")
            
    if not generated_embeddings:
        return None
    return np.mean(generated_embeddings, axis=0)

def load_references_from_db():
    """
    Loads all authorized facial embeddings from the database into memory.
    NOW SIMULATES A ROBUST, AVERAGED ENROLLMENT.
    """
    references = {}
    # --- DB Connection Logic would go here (Uncomment and configure later) ---
    # try:
    #     conn = psycopg2.connect(database="your_db", user="your_user", password="your_password", host="your_host", port="5432")
    #     cursor = conn.cursor()
    #     QUERY = """
    #         SELECT nombre, embedding_facial FROM Persona
    #         WHERE tipo = 'Resident' OR (tipo = 'Visitor' AND NOW() BETWEEN acceso_valido_desde AND acceso_valido_hasta)
    #     """
    #     cursor.execute(QUERY)
    #     for name, embedding_bytes in cursor.fetchall():
    #         # Assuming embedding is stored as float64 bytes
    #         references[name] = np.frombuffer(embedding_bytes, dtype=np.float64)
    #     cursor.close()
    #     conn.close()
    # except Exception as e:
    #     print(f"Database connection failed: {e}")
    #     print("Continuing with simulation data only.")
    
    # --- Robust Simulation ---
    
    try:
        # Define multiple images for the reference person
        reference_image_paths = [
            "Modules/Facial Recognition/photos/test/Abdullah_Gul/Abdullah_Gul_0001.jpg",
            "Modules/Facial Recognition/photos/test/Abdullah_Gul/Abdullah_Gul_0003.jpg",
            "Modules/Facial Recognition/photos/test/Abdullah_Gul/Abdullah_Gul_0004.jpg",
            "Modules/Facial Recognition/photos/test/Abdullah_Gul/Abdullah_Gul_0005.jpg",
            "Modules/Facial Recognition/photos/test/Abdullah_Gul/Abdullah_Gul_0006.jpg",
            "Modules/Facial Recognition/photos/test/Abdullah_Gul/Abdullah_Gul_0007.jpg",
            "Modules/Facial Recognition/photos/test/Abdullah_Gul/Abdullah_Gul_0008.jpg",
            "Modules/Facial Recognition/photos/test/Abdullah_Gul/Abdullah_Gul_0009.jpg",
            "Modules/Facial Recognition/photos/test/Abdullah_Gul/Abdullah_Gul_0010.jpg",
            "Modules/Facial Recognition/photos/test/Abdullah_Gul/Abdullah_Gul_0011.jpg",
            "Modules/Facial Recognition/photos/test/Abdullah_Gul/Abdullah_Gul_0012.jpg",
            "Modules/Facial Recognition/photos/test/Abdullah_Gul/Abdullah_Gul_0013.jpg",
            "Modules/Facial Recognition/photos/test/Abdullah_Gul/Abdullah_Gul_0014.jpg",
            "Modules/Facial Recognition/photos/test/Abdullah_Gul/Abdullah_Gul_0015.jpg",
            "Modules/Facial Recognition/photos/test/Abdullah_Gul/Abdullah_Gul_0016.jpg",
            "Modules/Facial Recognition/photos/test/Abdullah_Gul/Abdullah_Gul_0017.jpg",
            "Modules/Facial Recognition/photos/test/Abdullah_Gul/Abdullah_Gul_0018.jpg",
            "Modules/Facial Recognition/photos/test/Abdullah_Gul/Abdullah_Gul_0019.jpg"
        ]
        
        # Check if reference images exist
        existing_paths = [p for p in reference_image_paths if os.path.exists(p)]
        if not existing_paths:
            print(f"Warning: No reference images found for Abdullah Gul in Modules/Facial Recognition/photos/test/Abdullah_Gul/. Cannot create reference.")
            return references  # Return empty if no refs found

        print(f"Generating robust reference signature using {len(existing_paths)} images...")
        avg_embedding = generate_average_embedding(existing_paths)
        
        if avg_embedding is not None:
            references["Abdullah Gul"] = avg_embedding
        else:
            print("Warning: Could not create a reference embedding.")
            
    except Exception as e:
        print(f"Simulation error during reference loading: {e}")
    
    print(f"Loaded {len(references)} facial references into memory.")
    return references

def find_best_match(live_embedding, references):
    """
    Compares a live embedding against all known references.
    """
    best_name = "Unknown"
    best_similarity = 0.0
    best_match_name = None
    
    for name, ref_embedding in references.items():
        live_np = np.asarray(live_embedding)
        ref_np = np.asarray(ref_embedding)

        dot_product = np.dot(live_np, ref_np)
        norm_live = np.linalg.norm(live_np)
        norm_ref = np.linalg.norm(ref_np)
        
        if norm_live > 0 and norm_ref > 0:
            similarity = dot_product / (norm_live * norm_ref)
        else:
            similarity = 0.0

        if similarity > best_similarity:
            best_similarity = similarity
            best_match_name = name
            
    if best_similarity >= SIMILARITY_THRESHOLD:
        best_name = best_match_name
        
    return best_name, best_similarity

# --- Main Application Logic ---
if __name__ == "__main__":
    known_references = load_references_from_db()
    
    if not os.path.exists(IMAGE_TO_TEST):
        print(f"Error: Test image not found at '{IMAGE_TO_TEST}'.")
    else:
        frame = cv2.imread(IMAGE_TO_TEST)
        if frame is None:
            print(f"Error: Could not read the image file '{IMAGE_TO_TEST}'. It might be corrupted or in an unsupported format.")
        else:
            try:
                print(f"Processing image: {IMAGE_TO_TEST}")
                if frame.size == 0:
                    print("Error: Image loaded but is empty.")
                else:
                    detected_faces = DeepFace.extract_faces(
                        img_path=frame.copy(),
                        detector_backend=DETECTOR_BACKEND,
                        enforce_detection=False
                    )
                    
                    if not detected_faces or all(face['confidence'] == 0 for face in detected_faces):
                        print("No faces detected in the image.")

                    processed_a_face = False
                    for face in detected_faces:
                        if face.get('confidence', 0) == 0:
                            continue

                        fa = face['facial_area']
                        if not all(k in fa for k in ['x', 'y', 'w', 'h']):
                            print("Warning: Detected face data is incomplete. Skipping.")
                            continue

                        x, y, w, h = fa['x'], fa['y'], fa['w'], fa['h']
                        if w <= 0 or h <= 0 or x < 0 or y < 0 or y + h > frame.shape[0] or x + w > frame.shape[1]:
                            print(f"Warning: Invalid face coordinates detected ({x},{y},{w},{h}). Skipping.")
                            continue
                             
                        cropped_face = frame[y:y+h, x:x+w]
                        if cropped_face.size == 0:
                            print("Warning: Cropped face area is empty. Skipping.")
                            continue
                            
                        try:
                            live_embedding_info = DeepFace.represent(
                                img_path=cropped_face,
                                model_name=FACIAL_MODEL,
                                detector_backend='skip',
                                enforce_detection=False
                            )

                            if not live_embedding_info:
                                print("Warning: DeepFace.represent returned empty list for cropped face.")
                                continue

                            live_embedding = live_embedding_info[0]['embedding']
                            
                            name, similarity = find_best_match(np.array(live_embedding), known_references)
                            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                            
                            text = f"{name} ({similarity*100:.1f}%)"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.7
                            thickness = 2

                            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                            if y > text_height + 15:
                                text_y = y - 10
                            else:
                                text_y = y + h + text_height + 10

                            image_width = frame.shape[1]
                            ideal_text_x = x
                            
                            if ideal_text_x + text_width > image_width - 10:
                                text_x = max(0, x + w - text_width)
                                if text_x + text_width > image_width - 10:
                                    text_x = image_width - text_width - 10
                            else:
                                text_x = max(ideal_text_x, 5)

                            cv2.rectangle(frame, (text_x, text_y - text_height - baseline),
                                          (text_x + text_width, text_y + baseline),
                                          (50, 50, 50), cv2.FILLED)
                            
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
                            processed_a_face = True
                            
                        except ValueError as ve:
                            print(f"Error representing cropped face at ({x},{y},{w},{h}): {ve}")
                        except Exception as e:
                            print(f"Could not process a detected face: {e}")

            except Exception as e:
                print(f"An error occurred during face detection or processing: {e}")
            
            if frame is not None and processed_a_face:
                print("Processing complete. Displaying results. Press any key to exit.")
                cv2.imshow("Security System - Test Result", frame)
                cv2.waitKey(0)
            elif frame is not None and not processed_a_face:
                print("Processing complete, but no faces were successfully processed to display.")
            
            cv2.destroyAllWindows()
