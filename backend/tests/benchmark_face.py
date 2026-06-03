import os
import sys
from pathlib import Path

# Dynamically locates the 'backend' root and injects it into the execution path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import time
import numpy as np
from sqlalchemy import text, select
from src.database.session import SessionLocal
from src.database.models import Person

NUM_DUMMY_RESIDENTS = 25000  # Analytical stress testing universe
NUM_QUERIES_TEST = 200      # Samples to calculate average latency

def generate_normalized_vector():
    """Generates a random 512-D L2-normalized vector (Valid ArcFace signature)."""
    vec = np.random.randn(512)
    return (vec / np.linalg.norm(vec)).tolist()

def run_performance_benchmark():
    db = SessionLocal()
    print(f"--- STARTING BIOMETRIC STRESS TESTING PROTOCOL (N = {NUM_DUMMY_RESIDENTS}) ---")
    
    try:
        # 1. Bulk population of the database
        print("Populating database with random 512-D vectors...")
        bulk_persons = []
        for i in range(NUM_DUMMY_RESIDENTS):
            bulk_persons.append(
                Person(full_name=f"Test Resident {i}", person_type="RESIDENT", face_embedding=generate_normalized_vector())
            )
        db.add_all(bulk_persons)
        db.commit()
        print("Database populated and HNSW index updated automatically.")

        # Generate the target query vector
        target_vector = generate_normalized_vector()

        # 2. BENCHMARK A: Legacy Algorithm (Linear Scan O(N))
        print(f"\nExecuting {NUM_QUERIES_TEST} queries simulating O(N) (Forcing linear scan)...")
        latencies_linear = []
        
        for _ in range(NUM_QUERIES_TEST):
            t0 = time.time()
            with db.begin():
                # Force Postgres to ignore the HNSW graph index to simulate legacy sequential scan behavior
                db.execute(text("SET LOCAL enable_indexscan = off;"))
                db.execute(text("SET LOCAL enable_bitmapscan = off;"))
                
                distance_col = Person.face_embedding.cosine_distance(target_vector).label("distance")
                stmt = select(Person.full_name, distance_col).where(Person.face_embedding.is_not(None)).order_by(distance_col).limit(1)
                db.execute(stmt).first()
            latencies_linear.append((time.time() - t0) * 1000) # Convert to ms

        avg_linear = sum(latencies_linear) / NUM_QUERIES_TEST

        # 3. BENCHMARK B: Optimized Algorithm (HNSW Graph Search O(log N))
        print(f"Executing {NUM_QUERIES_TEST} queries with HNSW O(log N) optimization...")
        latencies_hnsw = []
        
        for _ in range(NUM_QUERIES_TEST):
            t0 = time.time()
            with db.begin():
                # Re-enable standard query planners and tune the local search buffer (ef_search)
                db.execute(text("SET LOCAL enable_indexscan = on;"))
                db.execute(text("SET LOCAL enable_bitmapscan = on;"))
                db.execute(text("SET LOCAL hnsw.ef_search = 32;"))
                
                distance_col = Person.face_embedding.cosine_distance(target_vector).label("distance")
                stmt = select(Person.full_name, distance_col).where(Person.face_embedding.is_not(None)).order_by(distance_col).limit(1)
                db.execute(stmt).first()
            latencies_hnsw.append((time.time() - t0) * 1000)

        avg_hnsw = sum(latencies_hnsw) / NUM_QUERIES_TEST

        # 4. MANDATORY METRICS REPORT (Senior/Tech Lead Style)
        print("\n===============================================================================")
        print(" PERFORMANCE REPORT: BIOMETRIC VECTOR RETRIEVAL")
        print("===============================================================================")
        print(f" Universe of residents in database (N) : {NUM_DUMMY_RESIDENTS}")
        print(f" Average Latency - Linear Scan O(N)     : {avg_linear:.2f} ms")
        print(f" Average Latency - HNSW Graph O(log N)     : {avg_hnsw:.2f} ms")
        
        improvement_percentage = ((avg_linear - avg_hnsw) / avg_linear) * 100
        print(f" Net Compute Time Reduction        : {improvement_percentage:.1f}%")
        print("===============================================================================")

    finally:
        print("\nCleaning up stress testing records to keep the database pristine...")
        db.rollback()
        db.execute(text("DELETE FROM persons WHERE full_name LIKE 'Test Resident%';"))
        db.commit()
        db.close()
        print(" Database sanitized successfully.")

if __name__ == "__main__":
    run_performance_benchmark()