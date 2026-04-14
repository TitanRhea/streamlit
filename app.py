import cv2
import mediapipe as mp
import math
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import base64
import os
import time
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components

# --- Ρυθμίσεις Σελίδας ---
st.set_page_config(page_title="SignAI Web Hub", layout="wide")

# --- Πλαϊνό Μενού (Sidebar) ---
st.sidebar.title("SignAI Menu 🚀")
page = st.sidebar.radio("Select Mode:", ["Recognition Camera", "Avatar Voice Mode"])

# --- Έλεγχος Κατάστασης για τον Ήχο (Κάμερα) ---
if "spoken_word" not in st.session_state:
    st.session_state.spoken_word = ""

# --- Συνάρτηση για τον Ήχο της Κάμερας ---
def play_local_sound(word, voice):
    sound_map = {
        "KALIMERA": "kalimera",
        "EFHARISTO": "efharisto",
        "GEIA": "geia",
        "KALO MESIMERI": "kalo_mesimeri",
        "ONOMA": "poio.einai.to.onoma.sou"
    }
    base = sound_map.get(word, "")
    if base:
        filename = f"{base}.{voice.lower()}.wav"
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                data = f.read()
                b64 = base64.b64encode(data).decode()
                md = f"""
                    <audio autoplay="true">
                    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                    </audio>
                    """
                st.markdown(md, unsafe_allow_html=True)

# ==========================================
# 1Η ΣΕΛΙΔΑ: ΚΑΜΕΡΑ (Recognition)
# ==========================================
if page == "Recognition Camera":
    st.title("📷 Live Recognition Mode")
    
    # Εδώ μπαίνει η κλάση της κάμερας ακριβώς όπως την κλειδώσαμε
    class SignLanguageProcessor(VideoProcessorBase):
        def __init__(self):
            self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.4)
            self.current_word = "WAITING..."
            self.last_word_time = time.time()
            self.start_time = time.time()
            self.history = []

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            active_now = None
            if results.multi_hand_landmarks:
                # ... [Εδώ τρέχει η λογική των κανόνων σου] ...
                # (Την κρατάμε ίδια με την προηγούμενη έκδοση)
                pass 

            # (Για συντομία εδώ, βάλε όλο το σώμα της recv που φτιάξαμε)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    voice_choice = st.radio("Voice:", ["Female", "Male"], horizontal=True)
    webrtc_streamer(key="sign-camera", video_processor_factory=SignLanguageProcessor)
    st_autorefresh(interval=1500, key="camera_refresh")

# ==========================================
# 2Η ΣΕΛΙΔΑ: ΑΒΑΤΑΡ (Voice Mode)
# ==========================================
else:
    st.title("🤖 Avatar Voice Mode")
    st.markdown("Μίλησε στο μικρόφωνο για να δεις το Άβαταρ να νοηματίζει!")

    # ΔΙΑΒΑΣΜΑ ΤΟΥ HTML ΑΡΧΕΙΟΥ ΣΟΥ
    try:
        # Ψάχνουμε το αρχείο μέσα στον φάκελο avatar_files
        html_path = "avatar_files/index.html"
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                avatar_html = f.read()
            
            # Προβολή του Avatar στην οθόνη
            components.html(avatar_html, height=800, scrolling=True)
        else:
            st.error("Missing 'index.html' inside 'avatar_files' folder.")
    except Exception as e:
        st.error(f"Error loading avatar: {e}")
