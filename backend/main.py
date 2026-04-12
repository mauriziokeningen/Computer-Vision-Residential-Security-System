import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import multiprocessing
import time
import logging
import sys

from src.ingestion.stream import start_ingestion
from src.modules.face.inference import start_face_model
from src.modules.weapons.inference import start_weapon_model
from src.orchestrator.rules import start_orchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SystemMain")

def main() -> None:
    """Initializes and manages the concurrent processes for the security system."""
    logger.info("Starting TT2 Security System (Face + Weapon Detection Mode)...")

    processes = [
        multiprocessing.Process(target=start_orchestrator, name="Orchestrator_Process"),
        multiprocessing.Process(target=start_face_model, name="Face_Process"),
        multiprocessing.Process(target=start_weapon_model, name="Weapon_Process"),
        multiprocessing.Process(target=start_ingestion, name="Ingestion_Process")
    ]

    try:
        # Start processes sequentially to ensure ports bind correctly before pushing data
        for p in processes:
            p.start()
            time.sleep(1)

        logger.info("All systems online. Press Ctrl+C to shut down.")
        
        while any(p.is_alive() for p in processes):
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n")
        logger.info("Keyboard interrupt received. Initiating graceful shutdown...")
    
    finally:
        for p in processes:
            if p.is_alive():
                logger.info(f"Pidiendo cierre civilizado a {p.name}...")
                p.terminate()
                p.join(timeout=1.5) 
                
                if p.is_alive():
                    logger.warning(f"¡{p.name} atascado! Aplicando kill...")
                    p.kill()
                    p.join(timeout=1)
                    
        logger.info("System successfully shut down.")
        sys.exit(0)

if __name__ == '__main__':
    main()