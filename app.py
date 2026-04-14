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

st.sidebar.title("SignAI Menu 🚀")
page = st.sidebar.radio("Επίλεξε Λειτουργία:", ["Recognition Camera", "Avatar Voice Mode"])

if "spoken_word" not in st.session_state:
    st.session_state.spoken_word = ""

# --- Ήχος (Διορθωμένο για το Ευχαριστώ) ---
def play_local_sound(phrase, voice):
    gender = voice.lower()
    sound_map = {
        "καλημέρα": "kalimera",
        "ευχαριστώ": "efharisto", 
        "γεια": "geia",
        "καλό μεσημέρι": "kalo_mesimeri",
        "ποιο είναι το όνομά σου": "poio.einai.to.onoma.sou"
    }
    base = sound_map.get(phrase, "")
    if base:
        filenames = [f"{base}.{gender}.wav", f"efcharisto.{gender}.wav"]
        for filename in filenames:
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    data = f.read()
                    b64 = base64.b64encode(data).decode()
                    md = f"""<audio autoplay="true"><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>"""
                    st.markdown(md, unsafe_allow_html=True)
                break

# ==========================================
# 1Η ΣΕΛΙΔΑ: ΚΑΜΕΡΑ (Ο ΔΙΚΟΣ ΣΟΥ ΚΛΕΙΔΩΜΕΝΟΣ ΚΩΔΙΚΑΣ)
# ==========================================
if page == "Recognition Camera":
    st.title("📷 Live Recognition Mode")
    
    class SignLanguageProcessor(VideoProcessorBase):
        def __init__(self):
            self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
            self.current_word = "WAITING..."
            self.recording_started_at = 0
            self.recording = False
            self.word_candidates = []

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
                    mp.solutions.drawing_utils.draw_landmarks(img, lm, mp.solutions.hands.HAND_CONNECTIONS)
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
                    if palm and dist_thumb_pinky > 0.15 and dist_index_middle > 0.03: moutza = True

            if h_cnt == 1 and y_index_tip < 0.65 and not moutza and not idx: active_now = "καλό μεσημέρι"
            elif h_cnt >= 2: active_now = "ευχαριστώ"
            elif moutza: active_now = "γεια"
            elif idx and h_high: active_now = "καλημέρα"
            elif idx and h_chest: active_now = "ποιο είναι το όνομά σου"

            if active_now:
                if not self.recording:
                    self.recording = True
                    self.recording_started_at = time.time()
                    self.word_candidates = [active_now]
                else:
                    self.word_candidates.append(active_now)
                    self.current_word = active_now

            if self.recording:
                if (time.time() - self.recording_started_at) >= 1.4:
                    if self.word_candidates:
                        final = max(set(self.word_candidates), key=self.word_candidates.count)
                        self.current_word = final
                    self.recording = False

            cv2.rectangle(img, (0, h-70), (w, h), (0, 0, 0), -1)
            cv2.putText(img, f"WORD: {self.current_word}", (20, h-25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    voice_choice = st.radio("Φωνή:", ["Female", "Male"], horizontal=True)
    ctx = webrtc_streamer(key="sign-camera", mode=WebRtcMode.SENDRECV, video_processor_factory=SignLanguageProcessor)

    if ctx.state.playing:
        st_autorefresh(interval=1500, key="camera_refresh")
        if ctx.video_processor:
            current = ctx.video_processor.current_word
            if current != "WAITING..." and current != st.session_state.spoken_word:
                play_local_sound(current, voice_choice)
                st.session_state.spoken_word = current

# ==========================================
# 2Η ΣΕΛΙΔΑ: ΑΒΑΤΑΡ (ΜΕΣΩ IFRAME)
# ==========================================
else:
    st.title("🤖 Avatar Voice Mode")
    st.markdown("Πάτα το κουμπί '🎙️ Ενεργοποίηση' και μίλησε στο Άβαταρ!")

    # Εδώ το Streamlit καλεί απευθείας το site σου που δουλεύει τέλεια
    URL_TO_GITHUB_PAGES = "https://titanrhea.github.io/avatar-noimatiki/" 

    try:
        # Εμφάνιση του iframe σε μεγάλο μέγεθος για να χωράει άνετα το Άβαταρ
        components.iframe(URL_TO_GITHUB_PAGES, width=1200, height=850, scrolling=False)
    except Exception as e:
        st.error(f"Σφάλμα φόρτωσης Άβαταρ: {e}")
