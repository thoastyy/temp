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


# from streamlit_webrtc.sample_utils.turn import get_ice_servers

# ---------- LOAD MODEL ----------
model = TwoStreamLipNetInfer().model


st.title("Webcam Live Feed")
FRAME_WINDOW = st.image([])

# https://stackoverflow.com/questions/52503187/getting-error-videoiomsmf-async-readsample-call-is-failed-with-error-statu

def capture_video():
    capture = cv2.VideoCapture(cv2.CAP_DSHOW)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")

    # Define the codec and create VideoWriter object
    output_folder = "streamlit_preview2.mp4"
    videoWriter = cv2.VideoWriter(output_folder, fourcc, 30.0, (640, 480))

    while st.session_state.run_capture:
        ret, frame = capture.read()

        if ret:
            frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_disp)

            videoWriter.write(frame)

    # Release resources
    capture.release()
    videoWriter.release()
    cv2.destroyAllWindows()

    # capture = cv2.VideoCapture(cv2.CAP_DSHOW)
    # # fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    # fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")

    # # Define the codec and create VideoWriter object
    # output_folder = 'streamlit_preview2' + '.mp4'
    # videoWriter = None
    # if (run):
    #     videoWriter = cv2.VideoWriter(output_folder, fourcc, 30.0, (640,480))
    #     # out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))

    # while run:
    #     ret, frame = capture.read()

    #     if ret:
    #         # frame = cv2.flip(frame,0)
    #         frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         FRAME_WINDOW.image(frame_disp)

    #         # write the flipped frame
    #         cv2.imshow("video", frame)
    #         videoWriter.write(frame)
    #     else:
    #         break
    
    # Release everything if job is finished
    capture.release()
    if videoWriter is not None:
        videoWriter.release()
    cv2.destroyAllWindows()


# # Checkbox to start/stop the webcam feed
# run = st.checkbox("Run")

# # Display the live webcam feed
# FRAME_WINDOW = st.image([])
# capture_video(run)

# # After stopping, display the recorded video
# if not run:

#     st.write("Stopped")

#     col1, col2 = st.columns(2)
#     with col1:
#         if os.path.exists("streamlit_preview2.mp4"):
#             with open("streamlit_preview2.mp4", "rb") as video:
#                 video_bytes = video.read()
#                 st.video(video_bytes)
#         # with st.spinner("Splitting video into frames"):
#         #     video, img_p, files = load_video("streamlit_preview.mp4")
#         #     st.markdown("Frames Generated {}".format(files))
#         # with st.spinner("Generating face coordinates"):
#         #     coordinates = generate_lip_coordinates("samples")
#         #     st.markdown(f"Coordinates Extracted:\n{coordinates}")

#     with col2:
#         st.info("Ready to make prediction!")
#         # if st.button("Generate"):
#         #     with st.spinner("Generating prediction"):
#         #         y = model(video[None, ...].cuda(), coordinates[None, ...].cuda())
#         #         txt = ctc_decode(y[0])
#         #         st.text(txt[-1])


# Button to start/stop recording
if "run_capture" not in st.session_state:
    st.session_state.run_capture = False

if st.button("Start Recording"):
    st.session_state.run_capture = True
    capture_video()

if st.button("Stop Recording"):
    st.session_state.run_capture = False

# Display the recorded video
if os.path.exists("streamlit_preview2.mp4"):
    if st.button("Generate"):
        # st.video("streamlit_preview2.mp4")
        with st.spinner("Splitting video into frames"):
            video, img_p, files = load_video("streamlit_preview2.mp4")
            st.markdown("Frames Generated {}".format(files))
        with st.spinner("Generating face coordinates"):
            coordinates = generate_lip_coordinates("samples")
            st.markdown(f"Coordinates Extracted:\n{coordinates}")
        with st.spinner("Generating prediction"):
                y = model(video[None, ...].cuda(), coordinates[None, ...].cuda())
                txt = ctc_decode(y[0])
                st.text(txt[-1])




