
import cv2
import os
import numpy as np
import mediapipe as mp

# SETTINGS 
# Set your paths and parameters here

SOURCE_DATA_PATH = "set your source data path here" # Path to raw video data organized in folders by action
KEYPOINT_PATH = "set your keypoint data path here" # Path to save extracted keypoint .npy files (IMPP)
SEQUENCE_LENGTH = 60



# MediaPipe access
try:
    mp_holistic = mp.solutions.holistic
except AttributeError:
    raise ImportError(
        "Your MediaPipe installation is incompatible. "
        "Install a stable version with: pip install mediapipe==0.10.9"
    )

def extract_keypoints(results):
    """
    Extract left and right hand keypoints.
    Returns 126 features (63 per hand).
    """

    if results.left_hand_landmarks:
        left_hand = np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
    else:
        left_hand = np.zeros(63)

    if results.right_hand_landmarks:
        right_hand = np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
    else:
        right_hand = np.zeros(63)

    return np.concatenate([left_hand, right_hand])


def preprocess_data():

    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(f"{SOURCE_DATA_PATH} does not exist.")

    actions = [
        d for d in os.listdir(SOURCE_DATA_PATH)
        if os.path.isdir(os.path.join(SOURCE_DATA_PATH, d))
    ]

    if not actions:
        print("No action folders found.")
        return

    print(f"Found actions: {actions}")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        for action in actions:
            print(f"\nProcessing action: {action}")

            action_path = os.path.join(SOURCE_DATA_PATH, action)
            videos = [v for v in os.listdir(action_path) if v.endswith(".mp4")]

            if not videos:
                print(f"No videos found in {action}")
                continue

            keypoint_dir = os.path.join(KEYPOINT_PATH, action)
            os.makedirs(keypoint_dir, exist_ok=True)

            for video_name in videos:
                video_path = os.path.join(action_path, video_name)
                cap = cv2.VideoCapture(video_path)

                if not cap.isOpened():
                    print(f"Could not open {video_path}")
                    continue

                sequence_keypoints = []

                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()

                    if not ret:
                        print(f"Video ended early at frame {frame_num}")
                        break

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)

                    keypoints = extract_keypoints(results)
                    sequence_keypoints.append(keypoints)

                cap.release()

                # Pad if needed
                while len(sequence_keypoints) < SEQUENCE_LENGTH:
                    sequence_keypoints.append(np.zeros(126))

                sequence_keypoints = np.array(sequence_keypoints)

                npy_name = os.path.splitext(video_name)[0] + ".npy"
                npy_path = os.path.join(keypoint_dir, npy_name)
                np.save(npy_path, sequence_keypoints)

                print(f"Processed {video_name} -> Saved {npy_path}")

    print("\n--- Preprocessing Complete! ---")


if __name__ == "__main__":
    os.makedirs(KEYPOINT_PATH, exist_ok=True)
    preprocess_data()