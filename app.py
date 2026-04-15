import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import numpy as np
import os
import pickle
import base64
import time

# --- Ρυθμίσεις Σελίδας ---
st.set_page_config(page_title="SignAI Live", layout="wide")

MODEL_FILE = "sign_model.pkl"

if "spoken_word" not in st.session_state:
    st.session_state.spoken_word = ""

# --- Συνάρτηση Ήχου ---
def play_local_sound(phrase, voice="Female"):
    gender = voice.lower()
    sound_map = {
        "KALIMERA": "kalimera",
        "EFHARISTO": "efharisto",
        "GEIA": "geia",
        "KALO MESIMERI": "kalo_mesimeri",
        "ONOMA": "poio.einai.to.onoma.sou"
    }
    base = sound_map.get(phrase, "")
    if base:
        filename = f"{base}.{gender}.wav"
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                data = f.read()
                b64 = base64.b64encode(data).decode()
                md = f'<audio autoplay="true"><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>'
                st.markdown(md, unsafe_allow_html=True)

# --- AI Video Processor ---
class AIProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5
        )
        self.model = None
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, 'rb') as f:
                self.model = pickle.load(f)
        self.last_prediction = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        # Χαμηλή ανάλυση για να μην κολλάει το σύμπαν
        img_small = cv2.resize(img, (320, 240))
        rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        data_row = np.zeros(126).tolist()

        if results.multi_hand_landmarks and self.model:
            for i, hand_lms in enumerate(results.multi_hand_landmarks):
                if i >= 2: break
                start_idx = i * 63
                for j, lm in enumerate(hand_lms.landmark):
                    data_row[start_idx + j*3] = lm.x
                    data_row[start_idx + j*3 + 1] = lm.y
                    data_row[start_idx + j*3 + 2] = lm.z
            
            try:
                self.last_prediction = self.model.predict([data_row])[0]
            except:
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("📷 SignAI Live Recognition")
st.write("Το σύστημα χρησιμοποιεί τον εκπαιδευμένο εγκέφαλο από τα βίντεό σου!")

if not os.path.exists(MODEL_FILE):
    st.error("⚠️ Το αρχείο 'sign_model.pkl' λείπει από τον φάκελο!")
else:
    voice = st.radio("Επίλεξε Φωνή:", ["Female", "Male"], horizontal=True)
    ctx = webrtc_streamer(key="live", video_processor_factory=AIProcessor)

    if ctx.video_processor:
        res = ctx.video_processor.last_prediction
        if res and res != st.session_state.spoken_word:
            play_local_sound(res, voice)
            st.session_state.spoken_word = res
            time.sleep(1.5)
