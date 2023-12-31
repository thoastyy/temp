from pathlib import Path
import cv2
import streamlit as st
from datetime import datetime

# from streamlit_webrtc.sample_utils.turn import get_ice_servers


st.title("Webcam Live Feed")
run = st.checkbox("Run")
FRAME_WINDOW = st.image([])
# https://stackoverflow.com/questions/52503187/getting-error-videoiomsmf-async-readsample-call-is-failed-with-error-statu
capture = cv2.VideoCapture(cv2.CAP_DSHOW)
# fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")

# Define the codec and create VideoWriter object
output_folder = 'app_input/output-' + datetime.now().time().strftime("%H-%M-%S") + '.mp4'
videoWriter = None
if (run):
    videoWriter = cv2.VideoWriter(output_folder, fourcc, 30.0, (640,480))
    # out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))

while run:
    ret, frame = capture.read()

    if ret == True:
        # frame = cv2.flip(frame,0)
        frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_disp)

        # write the flipped frame
        cv2.imshow("video", frame)
        videoWriter.write(frame)
    else:
        break
else:
    st.write("Stopped")
    # Release everything if job is finished
    capture.release()
    if videoWriter is not None:
        videoWriter.release()
    cv2.destroyAllWindows()


# def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
#     img = frame.to_ndarray(format="bgr24")

#     # perform edge detection
#     img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

#     return av.VideoFrame.from_ndarray(img, format="bgr24")


# RECORD_DIR = Path("./records")
# RECORD_DIR.mkdir(exist_ok=True)


# def app():
#     if "prefix" not in st.session_state:
#         st.session_state["prefix"] = str(uuid.uuid4())
#     prefix = st.session_state["prefix"]
#     in_file = RECORD_DIR / f"{prefix}_input.flv"
#     out_file = RECORD_DIR / f"{prefix}_output.flv"

#     def in_recorder_factory() -> MediaRecorder:
#         return MediaRecorder(
#             str(in_file), format="flv"
#         )  # HLS does not work. See https://github.com/aiortc/aiortc/issues/331

#     def out_recorder_factory() -> MediaRecorder:
#         return MediaRecorder(str(out_file), format="flv")

#     webrtc_streamer(
#         key="record",
#         mode=WebRtcMode.SENDRECV,
#         rtc_configuration={"iceServers": get_ice_servers()},
#         media_stream_constraints={
#             "video": True,
#             "audio": True,
#         },
#         video_frame_callback=video_frame_callback,
#         in_recorder_factory=in_recorder_factory,
#         out_recorder_factory=out_recorder_factory,
#     )

#     if in_file.exists():
#         with in_file.open("rb") as f:
#             st.download_button(
#                 "Download the recorded video without video filter", f, "input.flv"
#             )
#     if out_file.exists():
#         with out_file.open("rb") as f:
#             st.download_button(
#                 "Download the recorded video with video filter", f, "output.flv"
#             )


# if __name__ == "__main__":
#     app()
