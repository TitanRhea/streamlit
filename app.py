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

# --- Session States για την Ουρά Ήχου ---
if "spoken_word" not in st.session_state:
    st.session_state.spoken_word = ""
if "last_audio_played" not in st.session_state:
    st.session_state.last_audio_played = 0
if "current_audio_delay" not in st.session_state:
    st.session_state.current_audio_delay = 0.0

# --- Ήχος (ΚΛΕΙΔΩΜΕΝΟΣ) ---
def play_local_sound(phrase, voice):
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
        filenames = [f"{base}.{gender}.wav"]
        if phrase == "EFHARISTO":
            filenames.append(f"efcharisto.{gender}.wav") 
            
        for filename in filenames:
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    data = f.read()
                    b64 = base64.b64encode(data).decode()
                    md = f"""
                        <div id="audio_{time.time()}">
                            <audio autoplay="true">
                                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                            </audio>
                        </div>
                    """
                    st.markdown(md, unsafe_allow_html=True)
                break

# ==========================================
# 1Η ΣΕΛΙΔΑ: ΚΑΜΕΡΑ (ΚΑΘΑΡΗ ΟΘΟΝΗ & ΙΕΡΑΡΧΙΑ)
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

            # Καταγραφή
            if active_now:
                if not self.recording:
                    self.recording = True
                    self.recording_started_at = time.time()
                    self.word_candidates = [active_now]
                else:
                    self.word_candidates.append(active_now)
            elif self.recording:
                # Αν πέσουν τα χέρια σου, βάζει "NONE" για να μην μπερδευτεί
                self.word_candidates.append("NONE")

            if self.recording:
                # 1.5 δευτερόλεπτο χρόνος καταγραφής
                if (time.time() - self.recording_started_at) >= 1.5:
                    if self.word_candidates:
                        # --- ΤΟ ΜΥΣΤΙΚΟ: Η ΑΠΟΛΥΤΗ ΙΕΡΑΡΧΙΑ ---
                        counts = {word: self.word_candidates.count(word) for word in set(self.word_candidates)}
                        
                        # ΕΔΩ ΗΤΑΝ ΤΟ ΛΑΘΟΣ (το c >= 3). Το έκανα c >= 1 !
                        valid_words = [w for w, c in counts.items() if c >= 1 and w != "NONE"]
                        
                        final = "NONE"
                        # Εδώ λέμε στην κάμερα: Αν είδες Όνομα, ΚΛΕΙΔΩΣΕ ΤΟ!
                        if "ONOMA" in valid_words:
                            final = "ONOMA"
                        # Αν δεν είδες Όνομα αλλά είδες Ευχαριστώ, ΚΛΕΙΔΩΣΕ ΤΟ (αγνόησε το Γεια)
                        elif "EFHARISTO" in valid_words:
                            final = "EFHARISTO"
                        elif "KALIMERA" in valid_words:
                            final = "KALIMERA"
                        elif "GEIA" in valid_words:
                            final = "GEIA"
                        elif "KALO MESIMERI" in valid_words:
                            final = "KALO MESIMERI"
                        
                        if final != "NONE":
                            self.speak_queue.append(final)
                            
                    self.recording = False
                    self.word_candidates = [] # Καθαρίζει τη μνήμη για το επόμενο νόημα!

            # ΚΑΘΑΡΗ ΟΘΟΝΗ!
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
            # ==========================================
            # ΕΔΩ ΛΥΝΕΤΑΙ Η ΟΥΡΑ: Ο ΔΥΝΑΜΙΚΟΣ ΧΡΟΝΟΣ
            # ==========================================
            if len(ctx.video_processor.speak_queue) > 0:
                now = time.time()
                # Περιμένει όσο του λέει η προηγούμενη λέξη
                if now - st.session_state.last_audio_played > st.session_state.current_audio_delay: 
                    word_to_speak = ctx.video_processor.speak_queue.pop(0)
                    
                    # Ορίζει πόσο χρόνο θα δώσει σε αυτή τη λέξη για να ακουστεί ολόκληρη!
                    durations = {
                        "KALIMERA": 1.5,
                        "EFHARISTO": 1.7,
                        "GEIA": 1.2,
                        "KALO MESIMERI": 2.2,
                        "ONOMA": 5.0  # ΕΔΩ ΕΒΑΛΑ ΤΟ 5.0 ΠΟΥ ΖΗΤΗΣΕΣ
                    }
                    st.session_state.current_audio_delay = durations.get(word_to_speak, 1.7)
                    
                    play_local_sound(word_to_speak, voice_choice)
                    st.session_state.last_audio_played = now

# ==========================================
# 2Η ΣΕΛΙΔΑ: ΑΒΑΤΑΡ (ΚΛΕΙΔΩΜΕΝΟ)
# ==========================================
else:
    st.title("🤖 Avatar Voice Mode")
    st.markdown("Πάτα το κουμπί '🎙️ Ενεργοποίηση' και μίλησε στο Άβαταρ!")

    URL_TO_GITHUB_PAGES = "https://titanrhea.github.io/avatar-noimatiki/" 

    try:
        components.iframe(URL_TO_GITHUB_PAGES, width=1200, height=850, scrolling=False)
    except Exception as e:
        st.error(f"Σφάλμα φόρτωσης Άβαταρ: {e}")
