

import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# MediaPipe Hands setup 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,  #looking for a maximum of two hands
    min_detection_confidence=0.3
)

DATA_DIR = 'set your data directory here' # <-- CHANGE THIS
MAX_HANDS = 2
FEATURES_PER_HAND = 42 # 21 landmarks * 2 coordinates (x, y)

data = []
labels = []

print("Starting data processing for 1 OR 2 hands...")

for dir_ in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_path):
        continue

    print(f'Processing class: {dir_}')
    for img_path in os.listdir(class_path):
        # final feature vector, padded with zeros
        padded_features = []
        
        full_img_path = os.path.join(class_path, img_path)
        img = cv2.imread(full_img_path)
        if img is None:
            print(f"Warning: Could not read image {full_img_path}. Skipping.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb) # attempts to detect hands.

        # <-- CHANGE: Process if ANY hands are detected
        if results.multi_hand_landmarks:
            # Collect all landmarks from all detected hands
            all_landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                all_landmarks.extend(hand_landmarks.landmark)

            # Normalize based on the landmarks present in the frame
            x_ = [lm.x for lm in all_landmarks]
            y_ = [lm.y for lm in all_landmarks]
            min_x, min_y = min(x_), min(y_)
            
            # Create the feature list for the detected hands
            temp_features = []
            for lm in all_landmarks:
                temp_features.append(lm.x - min_x)
                temp_features.append(lm.y - min_y)
            
            # <-- CHANGE: Pad the feature list with zeros to a fixed size
            num_features = len(temp_features)
            padded_features = temp_features + [0.0] * (MAX_HANDS * FEATURES_PER_HAND - num_features)
            
            data.append(padded_features)
            labels.append(dir_)
        else:
            print(f"Warning: No hands found in {full_img_path}. Skipping.")


print(f"Processing complete. Found {len(data)} valid samples.")

# Save the processed data and labels
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset saved to data.pickle")