

import cv2
import numpy as np
import os
import mediapipe as mp
from keras.models import load_model
from collections import deque
from PIL import ImageDraw, ImageFont, Image

# 1. CONFIGURATION


# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_PATH = r'D:\Data\modelpractice\newword_dynamic_model_data_opt_1.h5'
KEYPOINT_PATH = r'D:\Data\modelpractice\newwords_keypoint_data_words'

FONT_PATH = os.path.join(BASE_DIR, 'Manjari-Bold.ttf')

# --- TUNING ---
SEQUENCE_LENGTH = 60   
PREDICTION_INTERVAL = 5 
CONFIDENCE_THRESHOLD = 0.80

# --- UI SETTINGS ---
CAM_WIDTH, CAM_HEIGHT = 640, 480


# 2. LOAD RESOURCES


print("Loading CNN Model...")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load model. {e}")
    exit()

# --- AUTO-LOAD LABELS ---
if os.path.exists(KEYPOINT_PATH):
    LABELS = sorted([d for d in os.listdir(KEYPOINT_PATH) if os.path.isdir(os.path.join(KEYPOINT_PATH, d))])
    print(f"Loaded Labels: {LABELS}")
else:
    print("WARNING: 'keypoint_data' folder not found. Using default labels.")
    LABELS = ['no sign'] # Fallback

# Load Fonts
try:
    font_malayalam = ImageFont.truetype(FONT_PATH, 40) 
    font_ui = ImageFont.truetype("arial.ttf", 24)
except:
    font_malayalam = ImageFont.load_default()
    font_ui = ImageFont.load_default()


# 3. HELPER FUNCTIONS


def extract_keypoints(results):
    """Extracts exactly 126 features."""
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)

    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)

    return np.concatenate([lh, rh])

def draw_interface(frame, prediction, probability, status_color=(255, 255, 255)):
    """Draws text using Pillow."""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Banner
    draw.rectangle([(0,0), (CAM_WIDTH, 60)], fill=(40, 44, 52))

    # Text
    draw.text((20, 15), "Prediction:", font=font_ui, fill=(200, 200, 200))
    
    # Prediction Text
    draw.text((150, 5), prediction, font=font_malayalam, fill=status_color)

    # Confidence Bar
    if prediction not in ["...", "No Hands"]:
        bar_width = int(probability * 200)
        draw.rectangle([(CAM_WIDTH - 220, 20), (CAM_WIDTH - 20, 40)], outline=(255, 255, 255))
        draw.rectangle([(CAM_WIDTH - 220, 20), (CAM_WIDTH - 220 + bar_width, 40)], fill=(0, 255, 0))
        draw.text((CAM_WIDTH - 220, 45), f"{int(probability*100)}%", font=font_ui, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 4. MAIN LOOP

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

sequence = deque(maxlen=SEQUENCE_LENGTH)
current_pred = "..."
current_prob = 0.0
frame_count = 0

# Variables for Anti-Flicker logic
no_hands_frames = 0 
NO_HANDS_LIMIT = 2


with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # FIX: Initialize status_color every frame to ensure it resets when hands are detected again
        status_color = (255, 255, 255) 

        # 1. MediaPipe Processing
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        
        # 2. Check for Hands (Robust Logic)
        has_hands = (results.left_hand_landmarks is not None) or (results.right_hand_landmarks is not None)

        if has_hands:
            no_hands_frames = 0 # Reset counter because we see hands
            status_color = (255, 255, 255)

            # Draw Skeleton
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # 3. Extract & Append
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            # 4. Prediction Logic
            if len(sequence) == SEQUENCE_LENGTH and frame_count % PREDICTION_INTERVAL == 0:
                input_data = np.expand_dims(sequence, axis=0)
                
                res = model.predict(input_data, verbose=0)[0]
                idx = np.argmax(res)
                confidence = res[idx]

                if confidence > CONFIDENCE_THRESHOLD:
                    if idx < len(LABELS):
                        current_pred = LABELS[idx]
                        current_prob = confidence
                else:
                    current_pred = "..."
                    current_prob = 0.0
        
        else:
            # No hands detected 
            no_hands_frames += 1
            
            # Only switch to "No Hands" mode if we haven't seen hands for 10 frames
            if no_hands_frames > NO_HANDS_LIMIT:
                sequence.clear()
                current_pred = "No Hands"
                status_color = (255, 50, 50) # Red

        # 5. Draw UI 
        final_ui = draw_interface(frame, current_pred, current_prob, status_color)

        cv2.imshow('CNN Gesture Recognition', final_ui)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
