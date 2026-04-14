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

# --- Ήχος ---
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
        filenames = [f"{base}.{gender}.wav", f"efcharisto.{gender}.wav", f"efharisto.{gender}.wav"]
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
# 1Η ΣΕΛΙΔΑ: ΚΑΜΕΡΑ (ΜΕ ΤΗΝ ΤΕΛΕΙΑ ΛΟΓΙΚΗ ΣΟΥ)
# ==========================================
if page == "Recognition Camera":
    st.title("📷 Live Recognition Mode")
    
    class SignLanguageProcessor(VideoProcessorBase):
        def __init__(self):
            # Η αυθεντική σου ρύθμιση
            self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
            self.recording = False
            self.recording_started_at = 0
            self.word_candidates = []
            
            # Μεταβλητές για την οθόνη και τον ήχο (όπως στο τοπικό σου)
            self.current_reading = ""
            self.final_word = "WAITING..."
            self.speak_queue = []

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Μικραίνουμε την εικόνα ΜΟΝΟ για ταχύτητα επεξεργασίας (δεν φαίνονται οι τελείες)
            img = cv2.resize(img, (640, 480)) 
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            
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
                    # Οι τελείες δεν ζωγραφίζονται για ταχύτητα!
                    
                    y_wrist = lm.landmark[0].y
                    # ΚΛΕΙΔΩΜΕΝΟ: Οι δικές σου αυθεντικές συντεταγμένες (Επανέφερα το 0.85)
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

            # ΚΛΕΙΔΩΜΕΝΗ ΙΕΡΑΡΧΙΑ 
            if h_cnt == 1 and y_index_tip < 0.65 and not moutza and not idx: active_now = "KALO MESIMERI"
            elif h_cnt >= 2: active_now = "EFHARISTO"
            elif moutza: active_now = "GEIA"
            elif idx and h_high: active_now = "KALIMERA"
            elif idx and h_chest: active_now = "ONOMA"

            # ΚΑΤΑΓΡΑΦΗ ΟΠΩΣ ΣΤΟ ΤΟΠΙΚΟ (1.8 δευτερόλεπτα)
            if active_now:
                if not self.recording:
                    self.recording = True
                    self.recording_started_at = time.time()
                    self.word_candidates = [active_now]
                    self.current_reading = active_now
                else:
                    self.word_candidates.append(active_now)
                    self.current_reading = active_now

            if self.recording:
                if (time.time() - self.recording_started_at) >= 1.8:
                    if self.word_candidates:
                        # Κλείδωμα της λέξης!
                        final = max(set(self.word_candidates), key=self.word_candidates.count)
                        self.final_word = final
                        self.speak_queue.append(final) # Τη βάζουμε στην ουρά για να ακουστεί!
                    self.recording = False
                    self.current_reading = "" # Σταματάει να διαβάζει

            # ΓΡΑΦΙΚΑ ΟΘΟΝΗΣ (Ακριβώς όπως στο τοπικό σου!)
            cv2.rectangle(img, (0, h-70), (w, h), (0, 0, 0), -1)
            if self.recording:
                # Κόκκινα γράμματα όταν "διαβάζει"
                cv2.putText(img, f"READING: {self.current_reading}", (20, h-25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                # Πράσινα γράμματα όταν κλειδώσει
                cv2.putText(img, f"WORD: {self.final_word}", (20, h-25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    voice_choice = st.radio("Φωνή:", ["Female", "Male"], horizontal=True)
    
    ctx = webrtc_streamer(
        key="sign-camera", 
        mode=WebRtcMode.SENDRECV, 
        video_processor_factory=SignLanguageProcessor,
        async_processing=True 
    )

    if ctx.state.playing:
        st_autorefresh(interval=1000, key="camera_refresh") # Πιο γρήγορο ρολόι για να προλαβαίνει τον ήχο
        if ctx.video_processor:
            # Αν υπάρχει λέξη στην ουρά για να ακουστεί, την παίζει!
            if len(ctx.video_processor.speak_queue) > 0:
                word_to_speak = ctx.video_processor.speak_queue.pop(0)
                play_local_sound(word_to_speak, voice_choice)

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
