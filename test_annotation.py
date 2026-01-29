import os
import sys
from tkinter import Tk, filedialog
import pprint

# -------------------------------------------------
# IMPORT YOUR PIPELINE MODULE
# -------------------------------------------------
# If this code is in the SAME file, you can remove this import
# and directly call predict(...)
#
# Example:
# from pipeline import predict
#
# Adjust the import path if needed
# -------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# IMPORTANT:
# Replace `your_pipeline_file` with the filename
# where the code you pasted exists (without .py)
from TPOC import predict


# -------------------------------------------------
# IMAGE SELECTION POPUP
# -------------------------------------------------

def select_image_popup():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Select Image for Testing",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All Files", "*.*"),
        ],
    )

    root.destroy()
    return file_path


# -------------------------------------------------
# TEST RUNNER
# -------------------------------------------------

def run_test():
    print("\nOpening image selection popup...\n")
    image_path = select_image_popup()

    if not image_path:
        print("No image selected. Test aborted.")
        return

    if not os.path.exists(image_path):
        print("Selected file does not exist.")
        return

    print(f"Selected Image: {image_path}\n")
    print("Running prediction...\n")

    result = predict(image_path)

    print("=========== PREDICTION OUTPUT ===========\n")

    if not result or not result.get("objects"):
        print("No objects detected.")
        return

    # Pretty print full response
    pprint.pprint(result, indent=2)

    print("\n=========== OBJECT-WISE SUMMARY ===========\n")

    for obj in result["objects"]:
        print(
            f"ID: {obj['object_id']} | "
            f"Type: {obj['object_type']} | "
            f"Brand: {obj['brand']} | "
            f"Variant: {obj['variant']} | "
            f"Confidence: {obj['confidence']} | "
            f"OCR: {obj['ocr_text']}"
        )

    print("\n=========== BRAND COUNTS ===========\n")
    for brand, count in result["brand_counts"].items():
        print(f"{brand}: {count}")


# -------------------------------------------------
# MAIN
# -------------------------------------------------

if __name__ == "__main__":
    run_test()
