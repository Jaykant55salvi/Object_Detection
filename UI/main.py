# import cv2
# import streamlit as st
# import time
# import os
# import sys


# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# import TPOC

# def main():
#     st.set_page_config(page_title="Live Product Detection", layout="wide")
#     st.title("Live Webcam Product Detection")

#     start = st.button("Start Camera")
#     stop = st.button("Stop Camera")

#     if "run_cam" not in st.session_state:
#         st.session_state.run_cam = False

#     if start:
#         st.session_state.run_cam = True
#     if stop:
#         st.session_state.run_cam = False

#     frame_holder = st.empty()
#     result_holder = st.empty()

#     if not st.session_state.run_cam:
#         st.info("Click **Start Camera** to begin live detection")
#         return

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         st.error("Webcam not accessible")
#         return

#     while st.session_state.run_cam:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # ---- RUN FULL ANALYSIS PIPELINE ---- #
#         preds = TPOC.predict_from_frame(frame)

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_holder.image(frame_rgb, channels="RGB", width="stretch")
#         result_holder.json(preds)

#         time.sleep(0.15)  # throttle CPU/GPU load

#     cap.release()

# if __name__ == "__main__":
#     main()


import cv2
import streamlit as st
import time
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import TPOC

def main():
    st.set_page_config(page_title="Live Product Detection", layout="wide")
    st.title("Live Webcam Product Detection")

    col1, col2 = st.columns(2)

    with col1:
        start = st.button("▶ Start Camera")
    with col2:
        stop = st.button("⏹ Stop Camera")

    if "run_cam" not in st.session_state:
        st.session_state.run_cam = False

    if start:
        st.session_state.run_cam = True
    if stop:
        st.session_state.run_cam = False

    frame_holder = st.empty()
    result_holder = st.empty()

    if not st.session_state.run_cam:
        st.info("Click **Start Camera** to begin live detection")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Webcam not accessible")
        return

    while st.session_state.run_cam:
        ret, frame = cap.read()
        if not ret:
            break

        preds = TPOC.predict_from_frame(frame, visualize=True)

        annotated = preds.get("annotated_image")
        if annotated is not None:
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_holder.image(frame_rgb, channels="RGB", use_container_width=True)

        result_holder.json({
            "objects": preds["objects"],
            "brand_counts": preds["brand_counts"]
        })

        time.sleep(0.15)

    cap.release()

if __name__ == "__main__":
    main()
