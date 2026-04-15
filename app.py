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
st.set_page_config(page_title="SignAI Pro Live", layout="wide")

MODEL_FILE = "sign_model.pkl"

if "spoken_word" not in st.session_state:
    st.session_state.spoken_word = ""

# --- Συνάρτηση Αναπαραγωγής Ήχου ---
def play_local_sound(phrase, voice="Female"):
    gender = voice.lower()
    # Αντιστοίχιση των κλάσεων του μοντέλου με τα ονόματα των αρχείων ήχου
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
                # Δημιουργία HTML audio tag για αυτόματη αναπαραγωγή
                md = f'<audio autoplay="true"><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>'
                st.markdown(md, unsafe_allow_html=True)

# --- AI Video Processor ---
class AIProcessor(VideoProcessorBase):
    def __init__(self):
        # model_complexity=0 για μέγιστη ταχύτητα
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.model = None
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, 'rb') as f:
                self.model = pickle.load(f)
        self.last_prediction = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror effect
        
        # Επεξεργασία σε χαμηλή ανάλυση για αποφυγή καθυστερήσεων
        img_small = cv2.resize(img, (320, 240))
        rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        # 126 σημεία για 2 χέρια
        data_row = np.zeros(126).tolist()

        if results.multi_hand_landmarks and self.model:
            for i, hand_lms in enumerate(results.multi_hand_landmarks):
                if i >= 2: break
                # Προαιρετική σχεδίαση των σημείων (landmarks) στην οθόνη
                mp.solutions.drawing_utils.draw_landmarks(img, hand_lms, mp.solutions.hands.HAND_CONNECTIONS)
                
                start_idx = i * 63
                for j, lm in enumerate(hand_lms.landmark):
                    data_row[start_idx + j*3] = lm.x
                    data_row[start_idx + j*3 + 1] = lm.y
                    data_row[start_idx + j*3 + 2] = lm.z
            
            try:
                # Πρόβλεψη της κίνησης από το μοντέλο
                prediction = self.model.predict([data_row])[0]
                self.last_prediction = prediction
            except:
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Κύριο UI της Εφαρμογής ---
st.title("🖐️ SignAI: Live Recognition")
st.write("Η κάμερα αναγνωρίζει τις κινήσεις σας και τις μετατρέπει σε ήχο σε πραγματικό χρόνο.")

if not os.path.exists(MODEL_FILE):
    st.error(f"⚠️ Προσοχή: Το αρχείο '{MODEL_FILE}' δεν βρέθηκε. Παρακαλώ ανεβάστε το στον φάκελο της εφαρμογής.")
else:
    # Επιλογή φωνής
    voice = st.radio("Επιλογή Φωνής (Voice Selection):", ["Female", "Male"], horizontal=True)
    
    # Έναρξη του WebRTC Streamer
    ctx = webrtc_streamer(
        key="live-recognition",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=AIProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

    # Έλεγχος για αλλαγή στην πρόβλεψη και αναπαραγωγή ήχου
    if ctx.video_processor:
        res = ctx.video_processor.last_prediction
        if res and res != st.session_state.spoken_word:
            # Εμφάνιση της λέξης στην οθόνη
            st.success(f"Αναγνωρίστηκε: **{res}**")
            # Αναπαραγωγή του αντίστοιχου αρχείου ήχου
            play_local_sound(res, voice)
            # Ενημέρωση του session state για να μην επαναλαμβάνεται ο ήχος
            st.session_state.spoken_word = res
            time.sleep(1.0) # Μικρή καθυστέρηση για ομαλή ροή
