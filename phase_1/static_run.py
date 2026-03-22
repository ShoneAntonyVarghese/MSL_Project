

import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

#  SETTINGS 
CAMERA_INDEX = 1
MODEL_PATH = 'set your model path here' # <-- CHANGE THIS
FONT_SIZE = 50
MAX_HANDS = 2
FEATURES_PER_HAND = 42 # 21 landmarks * 2 coords

#  FONT & LABEL SETTINGS (IMP) 
# Please ensure this path is correct for your system.
FONT_PATH = 'set your font path here' # Using the path from your folder structure

labels_dict = {
    0: 'അ', 
    1: 'ആ', 
    2: 'ഇ', 
    3: 'ഉ', 
    4: 'ഋ', 
    5: 'എ', 
    6: 'ഒ', 
    7: 'ക', 
    8: 'ഖ', 
    9: 'ഗ', 
    10: 'ഘ', 
    11: 'ങ', 
    12: 'ച', 
    13: 'ഛ', 
    14: 'ജ', 
    15: 'ഞ', 
    16: 'ട', 
    17: 'ഠ', 
    18: 'ഡ', 
    19: 'ഢ', 
    20: 'ണ', 
    21: 'ത', 
    22: 'ഥ', 
    23: 'ദ', 
    24: 'ധ', 
    25: 'ന',
    26: 'പ', 
    27: 'ഫ', 
    28: 'ബ', 
    29: 'ഭ', 
    30: 'യ', 
    31: 'ര',
    32: 'ല',
    33: 'വ', 
    34: 'ശ',    
    35: 'ഷ',
    36: 'സ', 
    37: 'ഹ',
    38: 'ള',
    39: 'ഴ',
    40: 'റ'
} # Update with your gesture labels



# LOAD FONT WITH ERROR HANDLING 
try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    print(f"Successfully loaded font: {FONT_PATH}")
except (FileNotFoundError, IOError):
    print(f"!!! WARNING: Font file not found at '{FONT_PATH}'. Using default font.")
    font = ImageFont.load_default()

#  LOAD THE TRAINED MODEL 
try:
    with open(MODEL_PATH, 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()

#  SETUP CAMERA & MEDIAPIPE 
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

print("Starting real-time gesture recognition... Press 'Q' to quit.")

# MAIN LOOP 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        all_landmarks = []
        
       
        # We slice the list `[:MAX_HANDS]` to ensure we only process a maximum of 2 hands,
        # even if MediaPipe momentarily detects more. This prevents the ValueError.
        for hand_landmarks in results.multi_hand_landmarks[:MAX_HANDS]:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            all_landmarks.extend(hand_landmarks.landmark)

        # Feature creation logic remains the same 
        x_ = [lm.x for lm in all_landmarks]
        y_ = [lm.y for lm in all_landmarks]
        min_x, min_y = min(x_), min(y_)
        
        temp_features = []
        for lm in all_landmarks:
            temp_features.append(lm.x - min_x)
            temp_features.append(lm.y - min_y)
        
        num_features = len(temp_features)
        padded_features = temp_features + [0.0] * (MAX_HANDS * FEATURES_PER_HAND - num_features)

        # Make prediction
        prediction = model.predict([np.asarray(padded_features)])
        predicted_character = labels_dict.get(int(prediction[0]), f'Class {prediction[0]}')

        # Drawing logic remains the same 
        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
        x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        draw.text((x1, y1 - FONT_SIZE - 5), predicted_character, font=font, fill=(0, 255, 0))
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()