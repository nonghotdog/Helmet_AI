import streamlit as st
import numpy as np

from PIL import Image # เปิด image filem ที่ user upload

from ultralytics import YOLO

import cv2

import tempfile # เก็บ temp ไม่ใช้ก็ลบ
import os
import shutil 
import av # framerate

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import logging

RTC_CONFIGURATION = RTCConfiguration( {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}) #P2P STUN server
logging.getLogger("streamlit_webrtc").setLevel(logging.ERROR) # ลด log

modelList = {
    "v8s_640": "models/v8s_640.pt",
    "v8m_832": "models/v8m_832.pt",
}

INFERENCE_IMG_SIZE = 480

@st.cache_resource
def loadModel(path: str):
    st.info(f"Loading : {path}...", icon="📦")
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Error loading model {path}: {e}")
        return None

def sideBarConfig():
    allmodels = list(modelList.keys())
    defaultselect = [allmodels[0]] if allmodels else []

    modelName = st.sidebar.multiselect("Select 1||2 models", allmodels, default=defaultselect)
    if len(modelName) > 2:
        st.sidebar.warning("Only the first 2 models gonna be used (Too many models !!!)")
        modelName = modelName[:2]

    confidence = st.sidebar.slider("Select Model Confidence", 25, 100, 40) / 100

    sourceRadio = st.sidebar.radio("Select Source", ('Image', 'Video', 'Webcam (Real-time)'))

    return modelName, confidence, sourceRadio


def selectModel(modelName: list):
    if not modelName:
        return {}

    loadedModels = {}
    st.sidebar.info(f"Loading [{len(modelName)}] model...")
    for name in modelName:
        model_path = modelList[name]
        model = loadModel(model_path)
        if model is not None:
            loadedModels[name] = model
    
    if loadedModels:
        st.sidebar.success("loaded successfully")

    return loadedModels

# ตาม standard lib อยาก process ต้องสร้าง class ที่ inherit มาจาก VideoTransformerBase # realtime detection [1]
# https://github.com/whitphx/streamlit-webrtc
class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self, modelsN: dict, confidenceThreshold: float, imgsz: int): # constructor สร้างครั้งเดียวตอน webrtc_streamer โดนเรียก // init ใช้ตลอดตามค่าที่ set ไว้ // process = (models, confidence, imgsz) / framerate]
        self.models = modelsN
        self.confidenceThreshold = confidenceThreshold
        self.imgsz = imgsz 
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame: # โดนเรียกทุก frame
        img = frame.to_ndarray(format="bgr24") # มันจะแปลง frame เป็น np array ให้ (YOLO, cv2) ใช้ *** [case diff(format)] *** \start/
        processedFrames = []

        for name, model in self.models.items(): # 2 in 1 loop serial process ไม่ได้ parallel
            results = model(img, conf=self.confidenceThreshold, imgsz=self.imgsz, verbose=False)                   # 1.) model (X) process
            processedImage = results[0].plot()                                                                     # 2.) เอาผลลัพธ์มา plot() บน image (X) // return np array (BGR)
            cv2.putText(processedImage, f"[{name}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)        # 3.) แปะ text บอก model
            processedFrames.append(processedImage)                                                                 # 4.) เก็บ output model ใน list                      
        if len(processedFrames) == 2: # np.hstack ต่อ model
            finalFrame = np.hstack(processedFrames)
        elif len(processedFrames) == 1:
            finalFrame = processedFrames[0]
        else:
            finalFrame = img

        return av.VideoFrame.from_ndarray(finalFrame, format="bgr24") # แปลงกลับเป็น frame ให้ streamlit_webrtc *** [case diff(format)] ส่ง np array กลับ browser (display)ไม่ได้ *** \end/

def processAndDisplayDetection(frameOrImage, modelsN: dict, conf: float):
    processedImages = []
    inputFormat = frameOrImage 

    for name, model in modelsN.items():
        results = model(inputFormat, conf=conf, imgsz=INFERENCE_IMG_SIZE, verbose=False)                            # 1
        detectedImage = results[0].plot()                                                                           # 2
        detectedImageRGB = cv2.cvtColor(detectedImage, cv2.COLOR_BGR2RGB)
        cv2.putText(detectedImageRGB, f"[{name}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)           # 3
        processedImages.append(detectedImageRGB)                                                                    # 4
    if len(processedImages) == 2:
        final_image = np.hstack(processedImages)
        caption = "Side-by-Side"
    elif len(processedImages) == 1:
        final_image = processedImages[0]
        caption = f"Detection ({list(modelsN.keys())[0]})"
    else:
        return None, None 
        
    return final_image, caption

def handleImageSource(loadedModels: dict, confidence: float):
    st.write("### Select Source Option : :orange[[Image]]")
    uploadedFile = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploadedFile is not None:
        image = Image.open(uploadedFile)
        st.image(image, caption="Uploaded Image", width='stretch')

        if st.button("Start Detect"):
            with st.spinner('Detecting objects in image...'):
                final_image, caption = processAndDisplayDetection(image, loadedModels, confidence)
                if final_image is not None:
                    st.image(final_image, caption=caption, width='stretch', channels="RGB")
                    st.balloons()


def handleVideoSource(loadedModels: dict, confidence: float):
    st.write("### Select Source Option : :orange[[Video]]")
    uploadedFile = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov", "mkv"])

    if uploadedFile is not None:
        tempDir = tempfile.mkdtemp()
        tempFilePath = os.path.join(tempDir, uploadedFile.name)
        try:
            with open(tempFilePath, 'wb') as tfile:
                tfile.write(uploadedFile.read())
                
            cap = cv2.VideoCapture(tempFilePath)
            if st.button("Start Detect"):
                frame_placeholder = st.empty()
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress_bar = st.progress(0)
                frame_count = 0

                st.info("Processing video frame by frame...", icon="🎬")
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    finalFrameDisplay, caption = processAndDisplayDetection(frame, loadedModels, confidence)
                    if finalFrameDisplay is not None:
                        frame_placeholder.image(finalFrameDisplay, caption=caption, channels="RGB", width='stretch')
            
                    frame_count += 1
                    progress = frame_count/total_frames
                    progress_bar.progress(min(int(progress*100), 100))
                
                cap.release()
                progress_bar.empty()
                st.success("Video Detection Complete!")
        
        finally:
            shutil.rmtree(tempDir, ignore_errors=True)


def handleWebcamSource(loadedModels: dict, confidence: float):
    st.write("### Select Source Option : :orange[[Webcam]]")

    webrtc_streamer( key="real-time-detection", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION, video_processor_factory=lambda: YOLOVideoTransformer(loadedModels, confidence, INFERENCE_IMG_SIZE ),
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640}, 
                "height": {"ideal": 480},
                "frameRate": {"ideal": 30}
            }, 
            "audio": False
        },
        async_processing=True,
    )

st.set_page_config(page_title="Demo", layout="wide")

a,b = st.columns([0.7,0.3])
a.write("# :green[Helmet] or :red[Non_Helmet] Detector")
with b.expander("ดูวิธีใช้การงาน"):
    st.image("images/how-to.gif")
st.markdown("---")

modelName, confidence, sourceRadio = sideBarConfig()
loadedModels = selectModel(modelName)

if not loadedModels:
    st.warning("🚨 Please select at least one Model and ensure it loads correctly to proceed :)")

if sourceRadio == 'Image':
    handleImageSource(loadedModels, confidence)
elif sourceRadio == 'Video':
    handleVideoSource(loadedModels, confidence)
elif sourceRadio == 'Webcam (Real-time)':
    handleWebcamSource(loadedModels, confidence)