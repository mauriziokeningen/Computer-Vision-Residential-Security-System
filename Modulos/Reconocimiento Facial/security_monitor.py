import cv2
import numpy as np
from deepface import DeepFace
import psycopg2

# --- System Constants ---
SIMILARITY_THRESHOLD = 0.68  # Threshold for ArcFace. You may need to tune this.
FACIAL_MODEL = "ArcFace"
DETECTOR_BACKEND = "mtcnn"

def load_references_from_db():
    """
    Loads all authorized facial embeddings from the database into memory.
    """
    references = {} # Dictionary to store: {"Name": <numpy_embedding>}
    
    # --- DB Connection Logic ---
    # conn = psycopg2.connect(database="your_db", user="your_user", ...)
    # cursor = conn.cursor()
    
    # Query for residents or visitors with current access
    # QUERY = """
    #     SELECT nombre, embedding_facial FROM Persona
    #     WHERE tipo = 'Resident' OR (tipo = 'Visitor' AND NOW() BETWEEN acceso_valido_desde AND acceso_valido_hasta)
    # """
    # cursor.execute(QUERY)
    
    # for name, embedding_bytes in cursor.fetchall():
    #     references[name] = np.frombuffer(embedding_bytes, dtype=np.float64)
        
    # cursor.close()
    # conn.close()
    
    # --- Simulation (Remove this when DB is connected) ---
    try:
        ana_ref = DeepFace.represent("photos/ana/ana_1.jpg", model_name=FACIAL_MODEL, detector_backend=DETECTOR_BACKEND)[0]['embedding']
        references["Ana Torres"] = np.array(ana_ref)
    except FileNotFoundError:
        print("Warning: 'photos/ana/ana_1.jpg' not found for simulation. Running with no known faces.")
    except Exception as e:
        print(f"Simulation error: {e}")
    # --- End Simulation ---
    
    print(f"Loaded {len(references)} facial references into memory.")
    return references

def find_best_match(live_embedding, references):
    """
    Compares a live embedding against all known references.
    """
    best_name = "Unknown"
    best_similarity = 0.0
    
    for name, ref_embedding in references.items():
        # Cosine Similarity calculation
        similarity = np.dot(live_embedding, ref_embedding) / (np.linalg.norm(live_embedding) * np.linalg.norm(ref_embedding))
        
        if similarity > best_similarity:
            best_similarity = similarity
            
    if best_similarity >= SIMILARITY_THRESHOLD:
        best_name = name
        
    return best_name, best_similarity

# --- Main Application Loop ---
if __name__ == "__main__":
    
    # 1. Load known faces from DB into RAM
    known_references = load_references_from_db()
    
    # 2. Start video capture
    cap = cv2.VideoCapture(0) # Use 0 for webcam, or "rtsp://..." for an IP camera
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        try:
            # 3. Detect faces in the frame
            detected_faces = DeepFace.extract_faces(
                img_path=frame.copy(), 
                detector_backend=DETECTOR_BACKEND, 
                enforce_detection=False # Don't crash if no face is found
            )
            
            for face in detected_faces:
                if face['confidence'] == 0: continue # Skip empty detections

                x, y, w, h = face['facial_area'].values()
                cropped_face = frame[y:y+h, x:x+w]
                
                try:
                    # 4. Generate embedding for the live face
                    live_embedding_info = DeepFace.represent(
                        img_path=cropped_face,
                        model_name=FACIAL_MODEL,
                        detector_backend='skip', # 'skip' because it's already a cropped face
                        enforce_detection=False
                    )
                    live_embedding = live_embedding_info[0]['embedding']
                    
                    # 5. Compare against DB and make decision
                    name, similarity = find_best_match(np.array(live_embedding), known_references)
                    
                    # 6. Draw on-screen feedback
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) # Green for known, Red for unknown
                    text = f"{name} ({similarity*100:.1f}%)"
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # This is where you would log the "Event" to your database
                    # if name == "Unknown":
                    #    log_event_to_db("Acceso No Autorizado", camera_id=1, ...)

                except Exception as e:
                    pass # Ignore faces that can't be processed

        except Exception as e:
            pass # Ignore frames where no face is detected
            
        # Display the final image
        cv2.imshow("Security System Monitor", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()