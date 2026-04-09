import cv2
import cvzone
from ultralytics import YOLO
import os
import requests
import json
from db_handler import save_violation
from dotenv import load_dotenv
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "Media"
WEIGHTS_PATH = BASE_DIR / "Weights" / "best.pt"

#toggle debug
SHOW_DEBUG_WINDOWS = True   # True = show windows, False = batch mode 
DEBUG_WAIT_MS = 0           # 0 = windows stay up until kb interupt, 500 = window pops up for 500ms before closing


# plate recognizer API 
load_dotenv()
API_URL = os.getenv(
    "PLATE_RECOGNIZER_API_URL",
    "https://api.platerecognizer.com/v1/plate-reader/"
)
API_TOKEN = os.getenv("PLATE_RECOGNIZER_TOKEN")
if not API_TOKEN:
    raise RuntimeError("Missing PLATE_RECOGNIZER_TOKEN. Set it in .env or environment variables.")


def _plate_api_call(image_bgr, use_strict=True):
    """
    One Plate Recognizer API call.
    If use_strict=True, includes engine_config {"region": "strict"}.
    Returns parsed plate list (may be empty).
    """
    ok, image_jpg = cv2.imencode(".jpg", image_bgr)
    if not ok:
        print("Failed to encode image for plate recognition")
        return []

    files = {"upload": ("frame.jpg", image_jpg.tobytes(), "image/jpeg")}
    data = {"regions": "my"}  # Malaysia

    if use_strict:
        engine_config = {"region": "strict", "plates_per_vehicle": 1}
        data["config"] = json.dumps(engine_config)
    else:
        engine_config = {"plates_per_vehicle": 1}
        data["config"] = json.dumps(engine_config)

    headers = {"Authorization": f"Token {API_TOKEN}"}

    try:
        resp = requests.post(API_URL, headers=headers, data=data, files=files, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print("Plate API error:", e)
        return []

    result = resp.json()
    results = result.get("results", [])
    plates = []

    for r in results:
        plate_text = r.get("plate", "").upper()
        plate_score = r.get("score", 0.0)
        det_score = r.get("dscore", 0.0)
        plate_box = r.get("box", None)

        vehicle_box = None
        if isinstance(r.get("vehicle"), dict):
            vehicle_box = r["vehicle"].get("box", None)

        plates.append({
            "plate": plate_text,
            "score": plate_score,
            "dscore": det_score,
            "box": plate_box,
            "vehicle_box": vehicle_box,
            "raw": r
        })

    return plates


def recognize_plates_full(image_bgr):
    """
    Adaptive OCR:
    1) Try strict
    2) If no results, fallback to non-strict
    Returns: (plates, ocr_mode)
    """
    plates_strict = _plate_api_call(image_bgr, use_strict=True)
    if plates_strict:
        return plates_strict, "strict"

    plates_non_strict = _plate_api_call(image_bgr, use_strict=False)
    if plates_non_strict:
        return plates_non_strict, "non_strict_fallback"

    return [], "none"


# YOLO helmet detection script
yolo_model = YOLO(str(WEIGHTS_PATH))
class_labels = ["With Helmet", "Without Helmet"]


def detect_helmets(image_bgr, conf_thresh=0.4):
    """
    Runs YOLO on a BGR image and returns:
    - annotated image
    - list of detections: {class_name, conf, box}
    """
    results = yolo_model(image_bgr)
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            bw = x2 - x1
            bh = y2 - y1
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = class_labels[cls_id]

            if conf >= conf_thresh:
                detections.append({
                    "class_name": class_name,
                    "conf": conf,
                    "box": (x1, y1, x2, y2)
                })

                cvzone.cornerRect(image_bgr, (x1, y1, bw, bh))
                cvzone.putTextRect(
                    image_bgr,
                    f"{class_name} {conf:.2f}",
                    (x1, y1 - 10),
                    scale=0.8,
                    thickness=1,
                    colorR=(255, 0, 0)
                )

    return image_bgr, detections

# IoU helper
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea <= 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0

    return interArea / union

def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return (x1 <= px <= x2) and (y1 <= py <= y2)

# build box around violator to scan for plate and then capture with OCR
def build_violator_region(v_box, img_w, img_h):
    """
    Build a region from detected head/helmet bbox down to expected bike plate area.
    Scales with bbox + image resolution.
    """
    vx1, vy1, vx2, vy2 = v_box
    bw = vx2 - vx1
    bh = vy2 - vy1

    pad_x = max(int(3.0 * bw), int(0.12 * img_w))
    pad_up = max(int(1.2 * bh), int(0.05 * img_h))
    pad_down = max(int(10.0 * bh), int(0.55 * img_h))

    rx1 = vx1 - pad_x
    ry1 = vy1 - pad_up
    rx2 = vx2 + pad_x
    ry2 = vy2 + pad_down

    return clamp_box(rx1, ry1, rx2, ry2, img_w, img_h)


def is_rider_like(all_plates, violator_region, min_plate_score=0.30):
    """
    Returns True if there is at least one plausible plate/vehicle candidate
    near this violator region (center-inside check).
    """
    for p in all_plates:
        if p.get("score", 0.0) < min_plate_score:
            continue

        b = p.get("vehicle_box") if p.get("vehicle_box") else p.get("box")
        if not b:
            continue

        px1, py1, px2, py2 = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
        cx, cy = box_center((px1, py1, px2, py2))

        if point_in_box(cx, cy, violator_region):
            return True

    return False

# Process image
def process_image(image_path):
    image_path = str(Path(image_path).resolve())

    print(f"\nProcessing: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image")
        return None

    h, w, _ = img.shape

    t0 = time.time()
    annotated, detections = detect_helmets(img, conf_thresh=0.4)
    print("YOLO time:", round(time.time() - t0, 2), "s")

    violations = [d for d in detections if d["class_name"] == "Without Helmet"]

    if not violations:
        print("No Without Helmet class detected.")
        if SHOW_DEBUG_WINDOWS:
            win_name = os.path.basename(image_path)
            cv2.imshow(win_name, annotated)
            cv2.waitKey(DEBUG_WAIT_MS)
            cv2.destroyWindow(win_name)
        return {"image_path": image_path, "detections": detections, "violations_logged": []}

    # optional to process higher confidence first 
    violations.sort(key=lambda d: d["conf"], reverse=True)
    print(f"Without Helmet detected ({len(violations)}) – calling plate API on full image...")

    t1 = time.time()
    all_plates, ocr_mode = recognize_plates_full(img)
    print("OCR time:", round(time.time() - t1, 2), "s | OCR mode:", ocr_mode)

    IOU_THRESHOLD = 0.05
    MIN_PLATE_SCORE = 0.7

    violations_logged = []

    # store relative path safely
    pth = Path(image_path).resolve()
    try:
        rel_image_path = str(pth.relative_to(BASE_DIR))
    except ValueError:
        rel_image_path = str(pth)

    # de-dup per image to prevent double logging of same plate
    used_plates = set()

    for idx, violation in enumerate(violations, start=1):
        vx1, vy1, vx2, vy2 = violation["box"]

        violator_region = build_violator_region((vx1, vy1, vx2, vy2), w, h)
        rx1, ry1, rx2, ry2 = violator_region
        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

        rider_like = is_rider_like(all_plates, violator_region, min_plate_score=0.30)

        # if bounding box cannot find any plate assume pedestrian like and log as review 
        if not rider_like:
            status = "REVIEW"
            reasons = ["PEDESTRIAN_LIKE_CONTEXT"]
            if ocr_mode == "non_strict_fallback":
                reasons.append("OCR_FALLBACK_USED")

            review_reason = ";".join(sorted(set(reasons)))

            print(f"[V{idx}] Pedestrian-like context — logging UNKNOWN (no plate association).")

            t2 = time.time()
            save_violation(
                image_path=rel_image_path,
                helmet_status="Without Helmet",
                helmet_confidence=float(violation["conf"]),
                plate_number="UNKNOWN",
                plate_score=None,
                iou_with_vehicle=None,
                api_raw=None,
                ocr_mode=ocr_mode,
                status=status,
                review_reason=review_reason
            )
            print("DB time:", round(time.time() - t2, 2), "s")

            violations_logged.append({
                "helmet_conf": float(violation["conf"]),
                "plate": "UNKNOWN",
                "iou": None,
                "ocr_mode": ocr_mode,
                "status": status,
                "review_reason": review_reason
            })
            continue

        status = "CONFIRMED"
        reasons = []
        if ocr_mode == "non_strict_fallback":
            status = "REVIEW"
            reasons.append("OCR_FALLBACK_USED")

        best_match = None
        best_iou = 0.0
        inside_candidates = []
        outside_candidates = []

        for plate in all_plates:
            if plate.get("score", 0.0) < MIN_PLATE_SCORE:
                continue

            b = plate.get("vehicle_box") if plate.get("vehicle_box") else plate.get("box")
            if not b:
                continue

            px1, py1, px2, py2 = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
            assoc_region = (px1, py1, px2, py2)

            cx, cy = box_center(assoc_region)
            inside = point_in_box(cx, cy, violator_region)
            overlap = iou(violator_region, assoc_region)

            if inside:
                inside_candidates.append((plate, overlap))
            else:
                outside_candidates.append((plate, overlap))

        if inside_candidates:
            inside_candidates.sort(key=lambda t: (t[1], t[0].get("score", 0.0)), reverse=True)
            best_match, best_iou = inside_candidates[0]
        elif outside_candidates:
            outside_candidates.sort(key=lambda t: (t[1], t[0].get("score", 0.0)), reverse=True)
            best_match, best_iou = outside_candidates[0]

        # logging logic
        if best_match and best_iou >= IOU_THRESHOLD:
            plate_for_violation = best_match
            plate_number = plate_for_violation["plate"]
            plate_score = float(plate_for_violation.get("score", 0.0))
            iou_val = float(best_iou)
            api_raw = plate_for_violation.get("raw")

            # don't reuse same plate in one image (optional)
            if plate_number in used_plates:
                status = "REVIEW"
                reasons.append("DUPLICATE_PLATE_SAME_IMAGE")
            else:
                used_plates.add(plate_number)

            print(
                f"[V{idx}] Linked plate: {plate_number} "
                f"(score={plate_score:.2f}, IoU={iou_val:.3f})"
            )

            # draw plate bbox if present
            pb = plate_for_violation.get("box")
            if pb:
                bx1, by1, bx2, by2 = pb["xmin"], pb["ymin"], pb["xmax"], pb["ymax"]
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    plate_number,
                    (bx1, max(0, by1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        else:
            plate_number = "UNKNOWN"
            plate_score = None
            iou_val = None
            api_raw = None

            status = "REVIEW"
            if not all_plates:
                reasons.append("NO_PLATE_FOUND")
            else:
                reasons.append("WEAK_ASSOCIATION")

            print(f"[V{idx}] No suitable plate match — logging UNKNOWN (status=REVIEW).")

        review_reason = ";".join(sorted(set(reasons))) if reasons else None

        t2 = time.time()
        save_violation(
            image_path=rel_image_path,
            helmet_status="Without Helmet",
            helmet_confidence=float(violation["conf"]),
            plate_number=plate_number,
            plate_score=plate_score,
            iou_with_vehicle=iou_val,
            api_raw=api_raw,
            ocr_mode=ocr_mode,
            status=status,
            review_reason=review_reason
        )
        print("DB time:", round(time.time() - t2, 2), "s")

        violations_logged.append({
            "helmet_conf": float(violation["conf"]),
            "plate": plate_number,
            "iou": iou_val,
            "ocr_mode": ocr_mode,
            "status": status,
            "review_reason": review_reason
        })

    if SHOW_DEBUG_WINDOWS:
        win_name = os.path.basename(image_path)
        cv2.imshow(win_name, annotated)
        cv2.waitKey(DEBUG_WAIT_MS)
        cv2.destroyWindow(win_name)

    return {
        "image_path": image_path,
        "detections": detections,
        "violations_logged": violations_logged
    }

# process all images in folder
def process_folder(folder_path=None):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    folder_path = Path(folder_path) if folder_path else MEDIA_DIR

    if not folder_path.exists():
        print("Folder not found:", folder_path)
        return []

    image_files = [
        str(p) for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]

    if not image_files:
        print("No images found in", folder_path)
        return []

    all_results = []
    for p in image_files:
        result = process_image(p)
        all_results.append(result)

    if SHOW_DEBUG_WINDOWS:
        cv2.destroyAllWindows()

    return all_results


if __name__ == "__main__":
    #process_folder()
    process_image("Media/riders9.jpg")
