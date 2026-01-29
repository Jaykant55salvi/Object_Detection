# import os
# import cv2
# import torch
# import numpy as np
# from PIL import Image, ImageEnhance, ImageFilter
# import pytesseract
# from pytesseract import Output
# from transformers import CLIPProcessor, CLIPModel
# from ultralytics import YOLO
# import tempfile

# try:
#     import easyocr
# except Exception:
#     easyocr = None

# _EASYOCR_READER = None

# # ---------------- CONFIG ---------------- #

# PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# GENERIC_OBJECT_LABELS = [
#     "bottle",
#     "can",
#     "box",
#     "packet",
#     "food package",
#     "drink container"
# ]

# YOLO_MODEL_PATH = "/home/jaykantsalvi/Documents/projects/object detection/models/best.pt"
# yolo_model = YOLO(YOLO_MODEL_PATH)

# # ---------------- SAFETY HELPERS ---------------- #

# def ensure_rgb(bgr_img: np.ndarray):
#     """
#     Ensures image is valid 3-channel BGR for CLIP/OCR
#     """
#     if bgr_img is None or bgr_img.size == 0:
#         return None

#     if len(bgr_img.shape) == 2:
#         return cv2.cvtColor(bgr_img, cv2.COLOR_GRAY2BGR)

#     if bgr_img.shape[2] == 1:
#         return cv2.cvtColor(bgr_img, cv2.COLOR_GRAY2BGR)

#     if bgr_img.shape[2] > 3:
#         return bgr_img[:, :, :3]

#     return bgr_img

# # ---------------- CLIP ---------------- #

# class CLIPZeroShot:
#     def __init__(self):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model = CLIPModel.from_pretrained(
#             "openai/clip-vit-base-patch32"
#         ).to(self.device)
#         self.processor = CLIPProcessor.from_pretrained(
#             "openai/clip-vit-base-patch32"
#         )

#     def classify(self, image: Image.Image, prompts):
#         inputs = self.processor(
#             images=image,
#             text=prompts,
#             return_tensors="pt",
#             padding=True
#         ).to(self.device)

#         with torch.no_grad():
#             outputs = self.model(**inputs)

#         probs = outputs.logits_per_image.softmax(dim=1)
#         idx = probs.argmax().item()
#         return prompts[idx], probs[0][idx].item()

# # ---------------- OCR ---------------- #

# def extract_text(image: np.ndarray) -> str:
#     if image is None or image.size == 0:
#         return ""

#     global _EASYOCR_READER

#     def enhance(pil):
#         pil = ImageEnhance.Contrast(pil).enhance(1.2)
#         pil = ImageEnhance.Sharpness(pil).enhance(1.3)
#         pil = pil.filter(ImageFilter.UnsharpMask(radius=1, percent=150))
#         return pil

#     image = ensure_rgb(image)
#     if image is None:
#         return ""

#     pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     pil = enhance(pil)

#     if easyocr is not None:
#         if _EASYOCR_READER is None:
#             _EASYOCR_READER = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
#         text = " ".join(_EASYOCR_READER.readtext(np.array(pil), detail=0))
#         return text.lower()

#     return pytesseract.image_to_string(pil).lower()

# # ---------------- YOLO BRAND / VARIANT ---------------- #

# def parse_class_name(cls_name):
#     """
#     Parse class name into brand and variant based on annotation folder naming.

#     Examples:
#         coco_cola_diet   -> ("Coca-Cola", "diet")
#         coco_cola_zero   -> ("Coca-Cola", "zero")
#         fanta_regular    -> ("Fanta", "regular")
#         pepsi_regular    -> ("Pepsi", "regular")
#     """
#     parts = cls_name.split("_")

#     # If class name is too short, treat whole name as brand
#     if len(parts) < 2:
#         return cls_name.title(), "regular"

#     # Last part is always variant
#     variant = parts[-1].lower()

#     # Brand is everything before last part
#     brand_raw = "_".join(parts[:-1]).lower()

#     # Map known brands to proper capitalization
#     if brand_raw in ["coco_cola", "coca_cola"]:
#         brand = "Coca-Cola"
#     elif brand_raw == "fanta":
#         brand = "Fanta"
#     elif brand_raw == "pepsi":
#         brand = "Pepsi"
#     elif brand_raw == "sprite":
#         brand = "Sprite"
#     else:
#         brand = brand_raw.replace("_", " ").title()

#     return brand, variant


# def predict_brand_variant_multi(image_path):
#     results = yolo_model.predict(source=image_path, imgsz=640, conf=0.25, save=False)
#     detections = []

#     for r in results:
#         if r.obb is None:
#             continue

#         for i in range(len(r.obb.cls)):
#             cls_id = int(r.obb.cls[i].cpu().numpy())
#             conf = float(r.obb.conf[i].cpu().numpy())
#             cls_name = r.names[cls_id]

#             print("cls name______", cls_name)

#             brand, variant = parse_class_name(cls_name)
#             bbox = r.obb.xyxy[i].cpu().numpy().astype(int)

#             detections.append({
#                 "brand": brand,
#                 "variant": variant,
#                 "confidence": conf,
#                 "bbox": bbox
#             })

#     return detections

# # ---------------- MULTI-OBJECT PIPELINE ---------------- #

# _clip = CLIPZeroShot()

# def predict(image_path, sam_masks=None):
#     image_bgr = cv2.imread(image_path)
#     image_bgr = ensure_rgb(image_bgr)

#     if image_bgr is None:
#         return {"objects": [], "brand_counts": {}}

#     H, W = image_bgr.shape[:2]
#     objects = []
#     brand_counts = {}

#     yolo_objs = predict_brand_variant_multi(image_path)

#     if not yolo_objs:
#         yolo_objs = [{
#             "brand": None,
#             "variant": "regular",
#             "confidence": 0.0,
#             "bbox": np.array([0, 0, W, H])
#         }]

#     for idx, det in enumerate(yolo_objs, 1):
#         x1, y1, x2, y2 = det["bbox"]

#         # ---- CLAMP BBOX ---- #
#         x1 = max(0, min(x1, W - 1))
#         y1 = max(0, min(y1, H - 1))
#         x2 = max(1, min(x2, W))
#         y2 = max(1, min(y2, H))

#         if x2 <= x1 or y2 <= y1:
#             continue

#         crop = image_bgr[y1:y2, x1:x2]
#         crop = ensure_rgb(crop)

#         if crop is None or crop.size == 0:
#             continue

#         crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

#         # OCR
#         ocr_text = extract_text(crop)

#         # CLIP
#         label, obj_conf = _clip.classify(
#             crop_pil,
#             [f"a photo of a {x}" for x in GENERIC_OBJECT_LABELS]
#         )

#         final_conf = round(max(obj_conf, det["confidence"]), 2)

#         brand_key = det["brand"] or "unknown"
#         brand_counts[brand_key] = brand_counts.get(brand_key, 0) + 1

#         objects.append({
#             "object_id": f"obj_{idx}",
#             "object_type": label.replace("a photo of a ", ""),
#             "brand": det["brand"],
#             "variant": det["variant"],
#             "ocr_text": ocr_text,
#             "confidence": final_conf
#         })

#     return {
#         "objects": objects,
#         "brand_counts": brand_counts
#     }

# # ---------------- WEBCAM SAFE WRAPPER ---------------- #

# def predict_from_frame(frame_bgr, sam_masks=None):
#     with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
#         path = tmp.name
#         cv2.imwrite(path, frame_bgr)

#     try:
#         return predict(path, sam_masks)
#     finally:
#         try:
#             os.remove(path)
#         except OSError:
#             pass

import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import tempfile

try:
    import easyocr
except Exception:
    easyocr = None

_EASYOCR_READER = None

# ---------------- CONFIG ---------------- #

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

GENERIC_OBJECT_LABELS = [
    "bottle",
    "can",
    "box",
    "packet",
    "food package",
    "drink container"
]

YOLO_MODEL_PATH = "/home/jaykantsalvi/Documents/projects/object detection/models/best.pt"
yolo_model = YOLO(YOLO_MODEL_PATH)

# ---------------- SAFETY HELPERS ---------------- #

def ensure_rgb(bgr_img: np.ndarray):
    if bgr_img is None or bgr_img.size == 0:
        return None

    if len(bgr_img.shape) == 2:
        return cv2.cvtColor(bgr_img, cv2.COLOR_GRAY2BGR)

    if bgr_img.shape[2] == 1:
        return cv2.cvtColor(bgr_img, cv2.COLOR_GRAY2BGR)

    if bgr_img.shape[2] > 3:
        return bgr_img[:, :, :3]

    return bgr_img

# ---------------- DRAW HELPERS ---------------- #

def draw_annotation(image, bbox, text, color=(0, 255, 0)):
    x1, y1, x2, y2 = bbox

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    (w, h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )

    cv2.rectangle(
        image,
        (x1, y1 - h - 8),
        (x1 + w + 4, y1),
        color,
        -1
    )

    cv2.putText(
        image,
        text,
        (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

# ---------------- CLIP ---------------- #

class CLIPZeroShot:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def classify(self, image: Image.Image, prompts):
        inputs = self.processor(
            images=image,
            text=prompts,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = outputs.logits_per_image.softmax(dim=1)
        idx = probs.argmax().item()
        return prompts[idx], probs[0][idx].item()

# ---------------- OCR ---------------- #

def extract_text(image: np.ndarray) -> str:
    if image is None or image.size == 0:
        return ""

    global _EASYOCR_READER

    def enhance(pil):
        pil = ImageEnhance.Contrast(pil).enhance(1.2)
        pil = ImageEnhance.Sharpness(pil).enhance(1.3)
        pil = pil.filter(ImageFilter.UnsharpMask(radius=1, percent=150))
        return pil

    image = ensure_rgb(image)
    if image is None:
        return ""

    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil = enhance(pil)

    if easyocr is not None:
        if _EASYOCR_READER is None:
            _EASYOCR_READER = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        text = " ".join(_EASYOCR_READER.readtext(np.array(pil), detail=0))
        return text.lower()

    return pytesseract.image_to_string(pil).lower()

# ---------------- YOLO PARSING ---------------- #

def parse_class_name(cls_name):
    parts = cls_name.split("_")

    if len(parts) < 2:
        return cls_name.title(), "regular"

    variant = parts[-1].lower()
    brand_raw = "_".join(parts[:-1]).lower()

    if brand_raw in ["coco_cola", "coca_cola"]:
        brand = "Coca-Cola"
    elif brand_raw == "fanta":
        brand = "Fanta"
    elif brand_raw == "pepsi":
        brand = "Pepsi"
    elif brand_raw == "sprite":
        brand = "Sprite"
    else:
        brand = brand_raw.replace("_", " ").title()

    return brand, variant

def predict_brand_variant_multi(image_path):
    results = yolo_model.predict(source=image_path, imgsz=640, conf=0.25, save=False)
    detections = []

    for r in results:
        if r.obb is None:
            continue

        for i in range(len(r.obb.cls)):
            cls_id = int(r.obb.cls[i].cpu().numpy())
            conf = float(r.obb.conf[i].cpu().numpy())
            cls_name = r.names[cls_id]

            brand, variant = parse_class_name(cls_name)
            bbox = r.obb.xyxy[i].cpu().numpy().astype(int)

            detections.append({
                "brand": brand,
                "variant": variant,
                "confidence": conf,
                "bbox": bbox
            })

    return detections

# ---------------- MAIN PIPELINE ---------------- #

_clip = CLIPZeroShot()

def predict(image_path, visualize=False):
    image_bgr = cv2.imread(image_path)
    image_bgr = ensure_rgb(image_bgr)

    if image_bgr is None:
        return {"objects": [], "brand_counts": {}, "annotated_image": None}

    H, W = image_bgr.shape[:2]
    objects = []
    brand_counts = {}

    yolo_objs = predict_brand_variant_multi(image_path)

    if not yolo_objs:
        yolo_objs = [{
            "brand": None,
            "variant": "regular",
            "confidence": 0.0,
            "bbox": np.array([0, 0, W, H])
        }]

    for idx, det in enumerate(yolo_objs, 1):
        x1, y1, x2, y2 = det["bbox"]

        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(1, min(x2, W))
        y2 = max(1, min(y2, H))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image_bgr[y1:y2, x1:x2]
        crop = ensure_rgb(crop)
        if crop is None:
            continue

        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        ocr_text = extract_text(crop)
        label, obj_conf = _clip.classify(
            crop_pil,
            [f"a photo of a {x}" for x in GENERIC_OBJECT_LABELS]
        )

        final_conf = round(max(obj_conf, det["confidence"]), 2)
        object_type = label.replace("a photo of a ", "")
        brand_text = det["brand"] or "unknown"

        display_text = f"{object_type} | {brand_text} {det['variant']} | {final_conf}"

        if visualize:
            draw_annotation(image_bgr, (x1, y1, x2, y2), display_text)

        brand_counts[brand_text] = brand_counts.get(brand_text, 0) + 1

        objects.append({
            "object_id": f"obj_{idx}",
            "object_type": object_type,
            "brand": det["brand"],
            "variant": det["variant"],
            "ocr_text": ocr_text,
            "confidence": final_conf
        })

    return {
        "objects": objects,
        "brand_counts": brand_counts,
        "annotated_image": image_bgr if visualize else None
    }

def predict_from_frame(frame_bgr, visualize=False):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        path = tmp.name
        cv2.imwrite(path, frame_bgr)

    try:
        return predict(path, visualize=visualize)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
