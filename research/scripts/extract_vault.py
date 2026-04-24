import zipfile
import argparse
import logging
from pathlib import Path

# ==========================================
# MLOps EPIC 2: SURGICAL DATA EXTRACTION
# Domain: Residential Security (MVP)
# Architecture: Strict Binary Violence Classifier
# ==========================================

# 1. Configure Enterprise Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# 2. The Residential MVP Taxonomy
THREAT_VECTORS = {
    "A050", # punch/slap
    "A051", # kicking (person-on-person)
    "A052", # pushing
}

HARD_NEGATIVES = {
    "A024", # kicking something (DECOY: allows kids to play soccer)
    "A055", # hugging (DECOY: prevents wrestling false positives)
    "A058", # shaking hands
    "A053", # pat on back (DECOY: prevents pushing false positives)
    "A054", # point finger
    "A010", # clapping (DECOY: fast hand movement)
    "A023", # hand waving
    "A025", # reach into pocket
    "A008", # sit down
    "A009", # stand up
}

# Union operator to combine sets for the extraction filter
MVP_CLASSES = THREAT_VECTORS | HARD_NEGATIVES

def extract_surgical_subset(zip_path: Path, output_dir: Path):
    """
    Surgically extracts only the residential MVP target classes.
    Flattens the directory structure for easier PyTorch ingestion.
    """
    if not zip_path.exists():
        logger.error(f"Payload not found at: {zip_path}")
        raise FileNotFoundError(f"Missing archive: {zip_path}")

    # Ensure output directory exists (creates /raw if it doesn't)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Opening Vault Payload: {zip_path.name}")
    logger.info(f"Targeting {len(MVP_CLASSES)} Residential MVP classes...")

    extracted_count = 0
    skipped_count = 0

    try:
        with zipfile.ZipFile(zip_path, 'r') as archive:
            files_in_archive = archive.infolist()
            logger.info(f"Total files in archive: {len(files_in_archive)}")

            for file_info in files_in_archive:
                # 3. Security & Hygiene: Skip directories and OS junk files
                if file_info.is_dir() or "__MACOSX" in file_info.filename or file_info.filename.startswith("."):
                    continue

                filename = Path(file_info.filename).name

                # 4. The Surgical Filter
                if any(action_class in filename for action_class in MVP_CLASSES):
                    # Flatten extraction (forces files into the root of output_dir)
                    file_info.filename = filename 
                    archive.extract(file_info, path=output_dir)
                    extracted_count += 1
                    logger.info(f"Extracted: {filename}")
                else:
                    skipped_count += 1
                    
    except zipfile.BadZipFile:
        logger.error("Archive is corrupted. The wget/curl download was likely interrupted.")
        raise

    logger.info("="*50)
    logger.info("EXTRACTION PIPELINE COMPLETE")
    logger.info(f"Total Extracted: {extracted_count} target files")
    logger.info(f"Total Skipped:   {skipped_count} noise files")
    logger.info("="*50)


if __name__ == "__main__":
    # 5. SOTA Argument Parsing
    parser = argparse.ArgumentParser(description="Extract MVP classes from NTU Zip archives.")
    
    # Resolves paths dynamically relative to where the script is executed
    project_root = Path.cwd()
    default_zip = project_root / "data" / "raw" / "body"/ "NTU_RGB_D"/ "nturgbd_rgb_s001.zip"
    default_out = project_root / "data" / "raw" / "body"/ "NTU_RGB_D"

    parser.add_argument(
        "--zip-path", 
        type=Path, 
        default=default_zip,
        help="Path to the downloaded .zip file."
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=default_out,
        help="Directory where extracted .avi files will be saved."
    )

    args = parser.parse_args()

    # Execute Pipeline
    extract_surgical_subset(args.zip_path, args.output_dir)