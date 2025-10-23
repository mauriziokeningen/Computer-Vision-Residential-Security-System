import cv2
import numpy as np
from deepface import DeepFace
import psycopg2 # Or your specific database connector

# --- Model Constants ---
FACIAL_MODEL = "ArcFace"
DETECTOR_BACKEND = "mtcnn"

def generate_average_embedding(image_path_list):
    """
    Takes a list of image paths, generates an embedding for each,
    and returns the average embedding.
    """
    generated_embeddings = []
    for img_path in image_path_list:
        try:
            # DeepFace.represent handles detection, alignment, and embedding
            embedding_info = DeepFace.represent(
                img_path=img_path,
                model_name=FACIAL_MODEL,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True
            )
            generated_embeddings.append(embedding_info[0]['embedding'])
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    if not generated_embeddings:
        return None

    # Calculate the average of all successful embeddings
    average_embedding = np.mean(generated_embeddings, axis=0)
    return average_embedding

def save_to_db(name, person_type, embedding):
    """
    Saves the new person and their facial embedding into the database.
    """
    
    # --- DB Connection Logic ---
    # conn = psycopg2.connect(database="your_db", user="your_user", ...)
    # cursor = conn.cursor()
    
    # Convert the numpy array embedding to bytes for database storage
    embedding_bytes = embedding.tobytes()
    
    # Insert into the Persona table
    # QUERY = """
    #     INSERT INTO Persona (nombre, tipo, embedding_facial, fecha_registro)
    #     VALUES (%s, %s, %s, NOW())
    # """
    # cursor.execute(QUERY, (name, person_type, embedding_bytes))
    # conn.commit()
    # cursor.close()
    # conn.close()
    
    print(f"Person '{name}' enrolled successfully.")

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Collect photos for the new resident
    resident_photos = ["photos/ana/ana_1.jpg", "photos/ana/ana_2.jpg", "photos/ana/ana_3.jpg"]

    # 2. Generate their unique facial signature
    print("Generating facial signature...")
    facial_signature = generate_average_embedding(resident_photos)

    # 3. Save it to the database
    if facial_signature is not None:
        save_to_db(name="Ana Torres", person_type="Resident", embedding=facial_signature)
    else:
        print("Could not generate facial signature. Check image paths and quality.")