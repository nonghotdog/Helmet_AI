import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image
import numpy as np
import av
import cv2

st.set_page_config(
    page_title="AI Helmet Detection System",
    page_icon="🤖",
    layout="wide"
)

st.write("# AI-Based System for Motorcycle Rider Helmet-Wearing Detection 🏍️")

#sidebar
    #select model
model_choice = st.sidebar.selectbox(
    "## Model Selection",
    ("Model [1] (เร็ว)",
     "Model [2] (แม่นยำ)")
)
    #input opion
source_choice = st.sidebar.radio(
    "## Input Source",
    ("Upload File (Image/Video)",
     "Realtime Detection (Live)")
)

st.sidebar.markdown("---")