import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceProcessorService:
    def __init__(self):
        """
        Initializes the RetinaFace and ArcFace models via InsightFace.
        Loads the 'buffalo_l' model pack into memory.
        """
        print("[AI] Initializing FaceProcessorService...")

        # RUNTIME GRAPH OPTIMIZATION: We prepare the models immediately to avoid latency during the first inference call.
        cuda_options = {
            "cudnn_conv_algo_search": "DEFAULT",
            "arena_extend_strategy": "kNextPowerOfTwo"
        }

        # Initialize the FaceAnalysis app. 
        # Note: The first time this runs, it will download ~330MB of models into ~/.insightface/models/
        self.app = FaceAnalysis(
            name='buffalo_l', 
            allowed_modules=['detection', 'recognition'],
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            provider_options=[cuda_options, {}]
        )
        
        
        # Prepare the execution environment
        # ctx_id=0 attempts to use the first GPU (CUDA). ctx_id=-1 forces CPU.
        # det_size=(640, 640) is the standard input resolution for RetinaFace.
        self.app.prepare(
            ctx_id=0, 
            det_size=(320, 320),
        )
        
        print("[AI] Face models loaded into memory successfully.")
        dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8) # A black image to warm up the GPU and trigger any lazy initialization in the models.
        self.app.get(dummy_frame)  # Warmup pass to trigger any lazy initialization
        
        print("[AI] Face models loaded and GPU graphs compiled successfully.")

    def extract_face_embedding(self, image_bytes: bytes) -> np.ndarray:
        """
        Decodes raw image bytes, detects the face, aligns it, and extracts the 512-d embedding.
        
        Args:
            image_bytes (bytes): The raw image file loaded into memory.
            
        Returns:
            np.ndarray: A 512-dimensional vector representing the face.
            
        Raises:
            ValueError: If no faces or multiple faces are detected in the image.
        """
        # 1. Convert byte array to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # 2. Decode image into an OpenCV BGR matrix
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image bytes into a valid image matrix.")

        # 3. Pass the image through the AI pipeline (Detection -> Alignment -> Extraction)
        faces = self.app.get(img)

        # 4. Strict Validation: For enrollment, we demand exactly ONE face per image.
        if len(faces) == 0:
            raise ValueError("No face detected in the image.")
        if len(faces) > 1:
            raise ValueError(f"Multiple faces ({len(faces)}) detected. Image must contain exactly one face.")

        # 5. Extract and return the 512-dimensional embedding (float32 numpy array)
        master_face = faces[0]
        return master_face.embedding

    def calculate_master_vector(self, embeddings: list[np.ndarray]) -> np.ndarray:
        """
        Calculates the centroid of multiple facial embeddings and applies L2 normalization.
        This creates a highly robust 'Master Vector' for the identity.
        
        Args:
            embeddings (list[np.ndarray]): A list of 512-dimensional face vectors.
            
        Returns:
            np.ndarray: A single, L2-normalized 512-dimensional master vector.
            
        Raises:
            ValueError: If the input list is empty.
        """
        if not embeddings:
            raise ValueError("Cannot calculate master vector from an empty list of embeddings.")

        # 1. Convert the list of arrays into a single 2D matrix
        # Shape becomes (N, 512) where N is the number of photos
        embeddings_matrix = np.array(embeddings)

        # 2. Calculate the Centroid (Mean across the columns / axis 0)
        # We sum all values in each dimension and divide by N. Shape becomes (512,)
        centroid = np.mean(embeddings_matrix, axis=0)

        # 3. Calculate the L2 Norm (The geometric length of the centroid vector)
        # Adding a tiny epsilon (1e-10) prevents division by zero in case of an empty vector
        l2_norm = np.linalg.norm(centroid) + 1e-10

        # 4. L2 Normalization (Scale the vector back to the unit hypersphere)
        master_vector = centroid / l2_norm

        return master_vector