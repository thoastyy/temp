import streamlit as st
from st_pages import Page, show_pages, add_page_title

import os 
from demo import load_video, ctc_decode
from model import LipNet
import torch
import torch.nn as nn
import os
from model import LipNet

# ---------- LOAD MODEL ----------
opt = __import__('options')
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu    

model = LipNet()
model = model.cuda()
net = nn.DataParallel(model).cuda()

if(hasattr(opt, 'weights')):
    pretrained_dict = torch.load(opt.weights)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
    print('miss matched params:{}'.format(missed_params))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


# ----- STREAMLIT STUFF ----------
st.set_page_config(layout='wide')

with st.sidebar: 
    st.title('SilentSpeak')
    st.info('Upload your video or do live inference!')

show_pages(
    [
        Page("app.py", "Upload", "🏠"),
        Page("app_live.py", "Live", "🏠"),
    ]
)

st.title('LipNet Full Stack App') 

# Generating a list of options or videos 
options = os.listdir(os.path.join('app_input'))
selected_video = st.selectbox('Choose video', options)

col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('app_input', selected_video)
        print(file_path)
        # TODO: Some conditional so that mp4 files need not be converted.
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 streamlit_preview.mp4 -y')

        # Rendering inside of the app
        video = open('streamlit_preview.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)
        with st.spinner("Splitting video into frames"):
            video, img_p, files = load_video("streamlit_preview.mp4")
            st.markdown('Frames Generated {}'.format(files))
        


    with col2: 
        st.info('Ready to make prediction!')
        if st.button('Generate'):                
            
            y = model(video[None,...].cuda())
            txt = ctc_decode(y[0])
            print("++ see txt: ", txt)
            st.text(txt[-1])
