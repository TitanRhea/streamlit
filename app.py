import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from gtts import gTTS
import base64
import os

# Ρυθμίσεις Σελίδας
st.set_page_config(page_title="SignAI - Python Power", layout="centered")
st.title("🚀 SignAI: Python Web Edition")
st.subheader("Το AI σου τρέχει τώρα με την πλήρη ισχύ της Python!")

# Φόρτωση Μοντέλου (Βάλε το σωστό όνομα του αρχείου σου)
@st.cache_resource
def load_my_model():
    return load_model('model.h5')

model = load_my_model()
labels = ["Καλό Μεσημέρι", "Γεια", "Καλημέρα", "Ευχαριστώ", "Όνομα"]

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Λειτουργία Ήχου (gTTS - Υπέροχη Γυναικεία Φωνή)
def speak(text):
    tts = gTTS(text=text, lang='el')
    tts.save("speech.mp3")
    audio_file = open("speech.mp3", "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
    os.remove("speech.mp3")

# Κάμερα (Χρησιμοποιούμε το streamlit-webrtc για ζωντανή ροή)
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.sequence = []
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Εδώ μπαίνει ο κώδικας που είχες στην Python για το prediction
        # (Θα τον προσαρμόσουμε μόλις δούμε ότι το site σηκώνεται)
        return frame

webrtc_streamer(key="sign-ai", video_processor_factory=SignLanguageProcessor)
