from pathlib import Path
import cv2
import streamlit as st
from datetime import datetime

import streamlit as st
from st_pages import Page, show_pages
import os
from demo import load_video, ctc_decode
from utils.two_stream_infer import TwoStreamLipNetInfer
import os
from scripts.extract_lip_coordinates import generate_lip_coordinates



# ---------- LOAD MODEL ----------
model = TwoStreamLipNetInfer().model

st.set_page_config(layout="wide")

with st.sidebar:
    st.title("SilentSpeak")
    st.info("Upload your video or do live inference!")

show_pages(
    [
        Page("app.py", "Upload", "💾"),
        # Page("app_live.py", "Live", "🎥"),
        Page("app_webcam.py", "Record", "👄"),
    ]
)
st.title("Webcam Record and Inference")
FRAME_WINDOW = st.image([])

# https://stackoverflow.com/questions/52503187/getting-error-videoiomsmf-async-readsample-call-is-failed-with-error-statu

def capture_video():
    capture = cv2.VideoCapture(cv2.CAP_DSHOW)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    # Define the codec and create VideoWriter object
    output_folder = "streamlit_preview2.mp4"
    videoWriter = cv2.VideoWriter(output_folder, fourcc, 30.0, (640, 480))
    # videoWriter = cv2.VideoWriter(output_folder, fourcc, 30.0, (360, 288))  # Set resolution to (360, 288)


    while st.session_state.run_capture:
        ret, frame = capture.read()

        if ret:
            frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_disp)

            videoWriter.write(frame)

    capture.release()
    print("Capture released")
    if videoWriter is not None:
        videoWriter.release()
    cv2.destroyAllWindows()
    print("Windows destroyed")



# Button to start/stop recording
if "run_capture" not in st.session_state:
    st.session_state.run_capture = False

if st.button("Start Recording"):
    st.session_state.run_capture = True
    capture_video()

if st.button("Stop Recording"):
    st.session_state.run_capture = False

# Process video
if os.path.exists("streamlit_preview2.mp4"):
    os.system("ffmpeg -i streamlit_preview2.mp4 -s 360x288 streamlit_preview_resized.mp4 -y")

    if os.path.exists("streamlit_preview_resized.mp4"):
        if st.button("Generate"):
            # st.video("streamlit_preview2.mp4")

            with st.spinner("Splitting video into frames"):
                video, img_p, files = load_video("streamlit_preview_resized.mp4")
                st.markdown("Frames Generated {}".format(files))
            with st.spinner("Generating face coordinates"):
                coordinates = generate_lip_coordinates("samples")
                st.markdown(f"Coordinates Extracted:\n{coordinates}")
            with st.spinner("Generating prediction"):
                    y = model(video[None, ...].cuda(), coordinates[None, ...].cuda())
                    txt = ctc_decode(y[0])
                    st.text(txt[-1])




