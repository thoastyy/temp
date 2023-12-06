import streamlit as st
from st_pages import Page, show_pages
import os
from demo import load_video, ctc_decode
from utils.two_stream_infer import TwoStreamLipNetInfer
import os
from scripts.extract_lip_coordinates import generate_lip_coordinates

# ---------- LOAD MODEL ----------
model = TwoStreamLipNetInfer().model

# ----- STREAMLIT STUFF ----------
st.set_page_config(layout="wide")

with st.sidebar:
    st.title("SilentSpeak")
    st.info("Upload your video or do live inference!")

show_pages(
    [
        Page("app.py", "Upload", "💾"),
        Page("app_live.py", "Live", "🎥"),
        Page("app_webcam.py", "Record", "👄"),
    ]
)

st.title('GUI') 

# Generating a list of options or videos
options = os.listdir(os.path.join("app_input"))
selected_video = st.selectbox("Choose video", options)

col1, col2 = st.columns(2)

if options:
    # Rendering the video
    with col1:
        st.info("The video below displays the converted video in mp4 format")
        file_path = os.path.join("app_input", selected_video)
        print(file_path)
        # TODO: Some conditional so that mp4 files need not be converted.
        os.system(f"ffmpeg -i {file_path} -vcodec libx264 streamlit_preview.mp4 -y")

        # Rendering inside of the app
        video = open("streamlit_preview.mp4", "rb")
        video_bytes = video.read()
        st.video(video_bytes)
        with st.spinner("Splitting video into frames"):
            video, img_p, files = load_video("streamlit_preview.mp4")
            st.markdown("Frames Generated {}".format(files))
            coordinates = generate_lip_coordinates("samples")
            st.markdown(f"Coordinates Extracted:\n{coordinates}")

    with col2:
        st.info("Ready to make prediction!")
        if st.button("Generate"):
            y = model(video[None, ...].cuda(), coordinates[None, ...].cuda())
            txt = ctc_decode(y[0])
            print("++ see txt: ", txt)
            st.text(txt[-1])
