import os
from minio import Minio
from minio.error import S3Error
from datetime import datetime
from src.database.session import SessionLocal
from src.database.models import Alert

# --- MINIO CONFIGURATION ---
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "password123")
BUCKET_NAME = "incident-evidence"

def check_postgresql():
    print("\n" + "="*55)
    print(" AUDITING POSTGRESQL (Latest 5 Alerts) ")
    print("="*55)
    
    db = SessionLocal()
    try:
        alerts = db.query(Alert).order_by(Alert.created_at.desc()).limit(5).all()
        
        if not alerts:
            print(" [WARN] The alerts table is empty. No events registered.")
            return
            
        for alert in alerts:
            print(f"[{alert.created_at.strftime('%H:%M:%S')}] ID: {str(alert.id)[:8]}... | STATUS: {alert.status}")
            print(f"    -> Message: {alert.message}\n")
            
    except Exception as e:
        print(f" [ERROR] Critical error connecting to PostgreSQL: {e}")
    finally:
        db.close()

def check_minio():
    print("="*55)
    print(f" AUDITING MINIO (Bucket: '{BUCKET_NAME}') ")
    print("="*55)
    
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        
        if not client.bucket_exists(BUCKET_NAME):
            print(f" [WARN] The bucket '{BUCKET_NAME}' does not exist in MinIO.")
            return

        # Use recursive=True to reach the .jpg files inside the incident folders
        objects = client.list_objects(BUCKET_NAME, recursive=True)
        
        # Filter: only keep items that are not directories and have a valid size
        object_list = [obj for obj in objects if not obj.is_dir and obj.size is not None]
        
        # Sort by date
        object_list.sort(key=lambda x: x.last_modified or datetime.min, reverse=True)
        
        if not object_list:
            print(" [WARN] No physical images found inside the folders.")
            return
            
        print(f"Total images found: {len(object_list)}\nShowing latest 5:\n")
        for obj in object_list[:5]:
            size_kb = obj.size / 1024
            print(f" [IMAGE] {obj.object_name}")
            print(f"    -> Size: {size_kb:.2f} KB | Date: {obj.last_modified.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
    except S3Error as e:
        print(f" [ERROR] S3 permission or structural error: {e}")
    except Exception as e:
        print(f" [ERROR] Network error during audit: {e}")

if __name__ == "__main__":
    print("\n INITIATING TT2 SECURITY SYSTEM E2E DIAGNOSTIC...\n")
    check_postgresql()
    check_minio()
    print(" Diagnostic complete.\n")