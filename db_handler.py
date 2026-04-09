import mysql.connector
from datetime import datetime
import json
import os

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "helmetwatch"
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def save_violation(
    image_path: str,
    helmet_status: str,
    helmet_confidence: float,
    plate_number: str,
    plate_score: float,
    iou_with_vehicle: float,
    api_raw: dict,
    ocr_mode="unknown",
    status="CONFIRMED",
    review_reason=None
):
    """
    Insert one violation record into the 'violations' table.
    Requires DB columns:
      - status
      - review_reason
    """

    image_name = os.path.basename(image_path)
    detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        sql = """
            INSERT INTO violations
            (image_name, image_path, detection_time,
             helmet_status, helmet_confidence,
             plate_number, plate_score, iou_with_vehicle, api_raw, ocr_mode,
             status, review_reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        values = (
            image_name,
            image_path,
            detection_time,
            helmet_status,
            helmet_confidence,
            plate_number,
            plate_score,
            iou_with_vehicle,
            json.dumps(api_raw) if api_raw is not None else None,
            ocr_mode,
            status,
            review_reason
        )

        cursor.execute(sql, values)
        conn.commit()
        print("✅ Violation saved to MySQL.")

    except mysql.connector.Error as e:
        print(f"[DB] Error saving violation: {e}")

    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass
