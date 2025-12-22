import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

def extraer_landmarks(frame, hands=None):
    if hands is None:
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0]
    coords = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
    return coords
