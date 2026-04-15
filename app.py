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

# --- Session States (Μνήμη) ---
if "spoken_word" not in st.session_state:
    st.session_state.spoken_word = ""
if "last_audio_played" not in st.session_state:
    st.session_state.last_audio_played = 0
if "current_audio_html" not in st.session_state:
    st.session_state.current_audio_html = ""
# ΝΕΟ: Εδώ αποθηκεύουμε πόσα δευτερόλεπτα πρέπει να περιμένει για την ΚΑΘΕ λέξη
if "audio_delay" not in st.session_state:
    st.session_state.audio_delay = 0.0

# --- Ήχος ---
def get_audio_html(phrase, voice, timestamp):
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
        filenames = [f"{base}.{gender}.wav", f"efcharisto.{gender}.wav", f"efharisto.{gender}.wav"]
        for filename in filenames:
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    data = f.read()
                    b64 = base64.b64encode(data).decode()
                    return f'<audio id="audio_{timestamp}" autoplay="true" style="display:none;"><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>'
    return ""

# ==========================================
# 1Η ΣΕΛΙΔΑ: ΚΑΜΕΡΑ
# ==========================================
if page == "Recognition Camera":
    st.title("📷 Live Recognition Mode")
    st.markdown("*(Tip: Κάνε ένα κλικ στη λέξη 'Female' ή 'Male' παρακάτω για να επιτρέψεις στον ήχο να παίξει)*")
    
    class SignLanguageProcessor(VideoProcessorBase):
        def __init__(self):
            self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
            self.recording = False
            self.recording_started_at = 0
            self.word_candidates = []
            self.speak_queue = [] 

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            img = cv2.resize(img, (640, 480)) 
            img = cv2.flip(img, 1)
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False 
            results = self.hands.process(rgb)
            rgb.flags.writeable = True

            active_now = None
            h_cnt = 0
            palm, idx, moutza = False, False, False
            h_chin, h_high, h_chest = False, False, False
            y_index_tip = 1.0

            if results.multi_hand_landmarks:
                h_cnt = len(results.multi_hand_landmarks)
                for lm in results.multi_hand_landmarks:
                    y_wrist = lm.landmark[0].y
                    # ΚΛΕΙΔΩΜΕΝΟ 0.85
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

            if h_cnt == 1 and y_index_tip < 0.65 and not moutza and not idx: active_now = "KALO MESIMERI"
            elif h_cnt >= 2: active_now = "EFHARISTO"
            elif moutza: active_now = "GEIA"
            elif idx and h_high: active_now = "KALIMERA"
            elif idx and h_chest: active_now = "ONOMA"

            # Καταγραφή κινήσεων
            if active_now:
                if not self.recording:
                    self.recording = True
                    self.recording_started_at = time.time()
                    self.word_candidates = [active_now]
                else:
                    self.word_candidates.append(active_now)
            elif self.recording:
                self.word_candidates.append("NONE")

            if self.recording:
                # 1.5 δευτερόλεπτο χρόνος "λοκαρίσματος" της κίνησης, όπως στο VS Code
                if (time.time() - self.recording_started_at) >= 1.5:
                    if self.word_candidates:
                        final = max(set(self.word_candidates), key=self.word_candidates.count)
                        if final != "NONE":
                            self.speak_queue.append(final) 
                    self.recording = False

            # ΕΠΙΣΤΡΕΦΕΙ ΤΗΝ ΕΙΚΟΝΑ ΚΑΘΑΡΗ! (Χωρίς cv2.putText και cv2.rectangle)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    voice_choice = st.radio("Φωνή:", ["Female", "Male"], horizontal=True)
    
    ctx = webrtc_streamer(
        key="sign-camera", 
        mode=WebRtcMode.SENDRECV, 
        video_processor_factory=SignLanguageProcessor,
        async_processing=True 
    )

    if ctx.state.playing:
        st_autorefresh(interval=1000, key="camera_refresh") 
        
        if ctx.video_processor:
            if len(ctx.video_processor.speak_queue) > 0:
                now = time.time()
                # Η εφαρμογή περιμένει όσο ακριβώς διαρκεί η προηγούμενη λέξη!
                if now - st.session_state.last_audio_played >= st.session_state.audio_delay: 
                    word_to_speak = ctx.video_processor.speak_queue.pop(0)
                    
                    # --- Ο ΕΞΥΠΝΟΣ ΧΡΟΝΟΜΕΤΡΗΤΗΣ ΤΗΣ VS CODE ---
                    # Εδώ ορίζουμε πόσο θα περιμένει για την ΕΠΟΜΕΝΗ λέξη 
                    # βάσει του πόσο μεγάλη είναι αυτή που παίζει τώρα.
                    durations = {
                        "KALIMERA": 1.2,
                        "EFHARISTO": 1.2,
                        "GEIA": 1.0,
                        "KALO MESIMERI": 1.8,
                        "ONOMA": 3.0  # Η μεγάλη φράση παίρνει 3 δεύτερα!
                    }
                    st.session_state.audio_delay = durations.get(word_to_speak, 1.5)
                    
                    new_html = get_audio_html(word_to_speak, voice_choice, now)
                    if new_html:
                        st.session_state.current_audio_html = new_html
                        st.session_state.last_audio_played = now

        # Αυτό μένει ζωντανό για να παίζει ο ήχος μέχρι τέλους
        if st.session_state.current_audio_html:
            st.markdown(st.session_state.current_audio_html, unsafe_allow_html=True)

# ==========================================
# 2Η ΣΕΛΙΔΑ: ΑΒΑΤΑΡ
# ==========================================
else:
    st.title("🤖 Avatar Voice Mode")
    st.markdown("Πάτα το κουμπί '🎙️ Ενεργοποίηση' και μίλησε στο Άβαταρ!")

    URL_TO_GITHUB_PAGES = "https://titanrhea.github.io/avatar-noimatiki/" 

    try:
        components.iframe(URL_TO_GITHUB_PAGES, width=1200, height=850, scrolling=False)
    except Exception as e:
        st.error(f"Σφάλμα φόρτωσης Άβαταρ: {e}")
