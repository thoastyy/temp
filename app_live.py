import streamlit as st
from st_pages import Page, show_pages, add_page_title
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import queue
import os
import av
import cv2
from demo import ctc_decode, load_frames25
from models.LipNet import LipNet
import torch
import torch.nn as nn
import os
from models.LipNet import LipNet

# ---------- LOAD MODEL ----------
opt = __import__("options")
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

model = LipNet()
model = model.cuda()
net = nn.DataParallel(model).cuda()

if hasattr(opt, "weights"):
    pretrained_dict = torch.load(opt.weights)
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict.keys() and v.size() == model_dict[k].size()
    }
    missed_params = [
        k for k, v in model_dict.items() if not k in pretrained_dict.keys()
    ]
    print(
        "loaded params/tot params:{}/{}".format(len(pretrained_dict), len(model_dict))
    )
    print("miss matched params:{}".format(missed_params))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("++ last line if")
print("++ after if")


# ----- STREAMLIT STUFF ----------
st.set_page_config(layout="wide")

with st.sidebar:
    st.title("SilentSpeak")
    st.info("Upload your video or do live inference!")

show_pages(
    [
        Page("app.py", "Upload", "💾"),
        Page("app_live.py", "Live", "🎥"),
    ]
)

st.title('GUI') 

# result_queue = queue.Queue()
result_queue: "queue.Queue[str]" = queue.Queue()

frames25 = []
counter = 0


# Generating a list of options or videos
# def video_frame_callback(frame: av.VideoFrame):
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    txt = ""
    print("Counter: {} frames25: {}".format(counter.len(frames25)))
    if counter != 25:
        frames25.append(image)
        counter += 1
    elif counter == 25:
        output_frames = load_frames25(frames25)
        y = model(output_frames[None, ...].cuda())
        txt = ctc_decode(y[0])
        frames25 = []
        counter = 0

    result_queue.put(txt)
    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    # rtc_configuration={
    #     "iceServers": get_ice_servers(),
    #     "iceTransportPolicy": "relay",
    # },
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if webrtc_ctx.state.playing:
    print("++ WEBRTC IS PLAYING")
    labels_placeholder = st.empty()
    # NOTE: The video transformation with object detection and
    # this loop displaying the result labels are running
    # in different threads asynchronously.
    # Then the rendered video frames and the labels displayed here
    # are not strictly synchronized.
    while True:
        print("++ in while loop")
        result = result_queue.get()
        labels_placeholder.text(result)
        print("display text")
        # if not result_queue.empty():
        # else:
        # print("Queue is empty.")
