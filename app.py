import cv2
import mediapipe as mp
import math
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# --- Ρυθμίσεις Σελίδας ---
st.set_page_config(page_title="SignAI Web", layout="centered")
st.title("🚀 SignAI: Python Web Edition")
st.markdown("Κάνε τις κινήσεις σου αργά και καθαρά μπροστά στην κάμερα.")

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# --- Κλάση Επεξεργασίας Βίντεο ---
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.current_word = "ΑΝΑΜΟΝΗ..."

    def recv(self, frame):
        # Παίρνουμε την εικόνα από την κάμερα του χρήστη
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
                
                # ΟΙ ΚΑΝΟΝΕΣ ΣΟΥ ΑΚΡΙΒΩΣ ΟΠΩΣ ΤΟΥΣ ΦΤΙΑΞΑΜΕ
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
            active_now = "ΚΑΛΟ ΜΕΣΗΜΕΡΙ"
        elif h_cnt >= 2: 
            active_now = "ΕΥΧΑΡΙΣΤΩ"
        elif moutza: 
            active_now = "ΓΕΙΑ"
        elif idx and h_high: 
            active_now = "ΚΑΛΗΜΕΡΑ"
        elif idx and h_chest: 
            active_now = "ΟΝΟΜΑ"

        # Ενημέρωση λέξης
        if active_now:
            self.current_word = active_now

        # Σχεδίαση UI πάνω στο βίντεο
        cv2.rectangle(img, (0, h-70), (w, h), (0, 0, 0), -1)
        cv2.putText(img, f"ΛΕΞΗ: {self.current_word}", (20, h-25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Εκκίνηση Κάμερας ---
webrtc_streamer(key="sign-language-app", video_processor_factory=SignLanguageProcessor)
