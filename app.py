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

# --- Ρυθμίσεις Σελίδας ---
st.set_page_config(page_title="SignAI Web", layout="centered")
st.title("🚀 SignAI: Python Web Edition")
st.markdown("Κάνε τις κινήσεις σου με φυσικότητα! Το σύστημα τώρα διαβάζει γρήγορα και με ελαστικότητα.")

# --- Έλεγχος Κατάστασης ---
if "spoken_word" not in st.session_state:
    st.session_state.spoken_word = ""

# --- Συνάρτηση Αναπαραγωγής Δικών σου Αρχείων ---
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

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# --- Κλάση Επεξεργασίας Βίντεο ---
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.4)
        self.current_word = "WAITING..."
        self.last_word_time = time.time()
        self.start_time = time.time() 
        self.history = [] 

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(rgb)

        active_now = None
        h_cnt = 0
        palm, idx, moutza = False, False, False
        h_chin, h_high, h_chest = False, False, False
        y_index_tip = 1.0

        if results.multi_hand_landmarks:
            h_cnt = len(results.multi_hand_landmarks)
            for lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)
                
                # ΟΙ ΚΑΝΟΝΕΣ ΣΟΥ
                y_wrist = lm.landmark[0].y
                
                if y_wrist < 0.50: h_chin = True 
                elif y_wrist < 0.85: h_high = True
                else: h_chest = True

                up = 0
                if lm.landmark[8].y < lm.landmark[6].y: up += 1
                if lm.landmark[12].y < lm.landmark[10].y: up += 1
                if lm.landmark[16].y < lm.landmark[14].y: up += 1
                if lm.landmark[20].y < lm.landmark[18].y: up += 1
                
                if up >= 3: palm = True
                elif up == 1: idx = True

                dist_thumb_pinky = math.hypot(lm.landmark[4].x - lm.landmark[20].x, lm.landmark[4].y - lm.landmark[20].y)
                dist_index_middle = math.hypot(lm.landmark[8].x - lm.landmark[12].x, lm.landmark[8].y - lm.landmark[12].y)
                y_index_tip = lm.landmark[8].y
                
                if palm and dist_thumb_pinky > 0.15 and dist_index_middle > 0.03:
                    moutza = True

        # Η ΙΕΡΑΡΧΙΑ ΣΟΥ 
        if h_cnt == 1 and y_index_tip < 0.65 and not moutza and not idx: 
            active_now = "KALO MESIMERI"
        elif h_cnt >= 2: 
            active_now = "EFHARISTO"
        elif moutza: 
            active_now = "GEIA"
        elif idx and h_high: 
            active_now = "KALIMERA"
        elif idx and h_chest: 
            active_now = "ONOMA"

        # --- Ο Νέος "Ελαστικός" Εγκέφαλος ---
        if time.time() - self.start_time > 2.0: # Ασπίδα 2 δευτερολέπτων στην αρχή
            self.history.append(active_now)
            
            # Κρατάμε μόνο 6 καρέ 
            if len(self.history) > 6:
                self.history.pop(0)
            
            # Πιο ελαστικό: Αν δει τη λέξη στα 3 από τα 6 καρέ, κλειδώνει!
            if active_now and self.history.count(active_now) >= 3:
                self.current_word = active_now
                self.last_word_time = time.time()
                
            # Καθαρισμός: Αν στα 6 καρέ τα 4 είναι άδεια (χέρια κάτω), πάει σε αναμονή
            elif self.history.count(None) >= 4:
                if time.time() - self.last_word_time > 1.2:
                    self.current_word = "WAITING..."

        # Σχεδίαση UI
        cv2.rectangle(img, (0, h-70), (w, h), (0, 0, 0), -1)
        cv2.putText(img, f"WORD: {self.current_word}", (20, h-25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI για Φωνή ---
voice_choice = st.radio("Voice:", ["Female", "Male"], horizontal=True)

# --- Εκκίνηση Κάμερας ---
ctx = webrtc_streamer(
    key="sign-language-app", 
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=SignLanguageProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Σύστημα Ήχου ---
if ctx.state.playing:
    # Δίνει χρόνο στον ήχο να ακουστεί ολόκληρος
    st_autorefresh(interval=1500, key="audiorefresh")
    
    if ctx.video_processor:
        current = ctx.video_processor.current_word
        
        if current == "WAITING...":
            st.session_state.spoken_word = ""
            
        elif current != "WAITING..." and current != st.session_state.spoken_word:
            play_local_sound(current, voice_choice)
            st.session_state.spoken_word = current
