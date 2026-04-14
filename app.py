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
st.markdown("Κάνε τις κινήσεις σου μπροστά στην κάμερα. Το σύστημα τώρα αντιδρά πιο γρήγορα!")

# --- Έλεγχος Κατάστασης ---
if "spoken_word" not in st.session_state:
    st.session_state.spoken_word = ""

# --- Συνάρτηση Αναπαραγωγής Δικών σου Αρχείων ---
def play_local_sound(word, voice):
    sound_map = {
        "καλημέρα": "kalimera",
        "ευχαριστώ": "efharisto",
        "γεια": "geia",
        "καλό μεσημέρι": "kalo_mesimeri",
        "όνομα": "poio.einai.to.onoma.sou"
    }
    base = sound_map.get(word.lower(), "")
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
        # Βάλαμε 0.5 για να πιάνει πιο εύκολα τα δύο χέρια στο "Ευχαριστώ"
        self.hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.current_word = "ΑΝΑΜΟΝΗ..."
        self.last_word_time = time.time()

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
            active_now = "καλό μεσημέρι"
        elif h_cnt >= 2: 
            active_now = "ευχαριστώ"
        elif moutza: 
            active_now = "γεια"
        elif idx and h_high: 
            active_now = "καλημέρα"
        elif idx and h_chest: 
            active_now = "όνομα"

        # ΛΟΓΙΚΗ ΤΑΧΥΤΗΤΑΣ & ΚΑΘΑΡΙΣΜΟΥ
        if active_now:
            # Χρειάζεται μόνο 0.5 δευτερόλεπτα για να καταλάβει τη λέξη
            if time.time() - self.last_word_time > 0.5:
                self.current_word = active_now
                self.last_word_time = time.time()
        else:
            # Αν δεν κάνεις καμία κίνηση για 2 δευτερόλεπτα, μηδενίζει
            if time.time() - self.last_word_time > 2.0:
                self.current_word = "ΑΝΑΜΟΝΗ..."
                self.last_word_time = time.time()

        # Σχεδίαση UI 
        cv2.rectangle(img, (0, h-70), (w, h), (0, 0, 0), -1)
        cv2.putText(img, f"ΛΕΞΗ: {self.current_word}", (20, h-25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI για Φωνή ---
voice_choice = st.radio("Επίλεξε Φωνή:", ["Female", "Male"], horizontal=True)

# --- Εκκίνηση Κάμερας ---
ctx = webrtc_streamer(
    key="sign-language-app", 
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=SignLanguageProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Το "Μαγικό" για να μην κολλάει ο ήχος (ΠΙΟ ΓΡΗΓΟΡΟ) ---
if ctx.state.playing:
    # Ανανεώνει τη σελίδα κάθε ΜΙΣΟ δευτερόλεπτο (500) αντί για ένα ολόκληρο
    st_autorefresh(interval=500, key="audiorefresh")
    
    if ctx.video_processor:
        current = ctx.video_processor.current_word
        
        # Αν η λέξη μηδενιστεί, ελευθερώνει τη μνήμη για να μπορεί να ξαναπεί την ίδια
        if current == "ΑΝΑΜΟΝΗ...":
            st.session_state.spoken_word = ""
            
        # Αν υπάρχει νέα λέξη, παίζει τον ήχο
        elif current != "ΑΝΑΜΟΝΗ..." and current != st.session_state.spoken_word:
            play_local_sound(current, voice_choice)
            st.session_state.spoken_word = current
