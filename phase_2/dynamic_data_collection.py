

import cv2
import os
import time
import mediapipe as mp


# 1. Path for the  data
DATA_PATH = 'path to your data directory' # <-- CHANGE THIS

# 2. List of signs you want to collect

# actions = ['അം', 
# 'അഃ', 
# 'ഈ', 
# 'ഊ', 
# 'ഏ',
# 'ഐ', 
# 'ഓ', 
# 'ഔ', 
# 'ഝ',  
# 'മ']

# CHANGE THIS according to your needs

actions = ['അമ്മ', 'ശരി', 'പണം', 'നന്ദി'] # <-- CHANGE THIS

# 3. Video count per sign
num_sequences = 25 # (e.g., 30 videos)

# 4. How long each video should be (in frames)
sequence_length = 60 # (e.g., 30 frames long)




# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def draw_styled_landmarks(image, results):
    # Draw hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

# Main data collection loop 
def collect_data():
    cap = cv2.VideoCapture(1)
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Use the Holistic model (x,y,z) for both hands
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # Loop through all actions (signs)
        for action in actions:
            print(f"\n--- Starting collection for: {action} ---")
            
            # Create the folder for this action
            action_path = os.path.join(DATA_PATH, action)
            os.makedirs(action_path, exist_ok=True)
            
            # Loop through all sequences (videos)
            for sequence in range(num_sequences):
                print(f"  Preparing to record sequence {sequence+1}/{num_sequences} for {action}")
                
                # Wait for user to press 's' to start recording
                while True:
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1) # Flip for selfie view
                    
                    # Process with MediaPipe (to show landmarks)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    draw_styled_landmarks(image, results)
                    
                    # Add text
                    text = f"PRESS 'S' TO START RECORDING: {action} - Video {sequence+1}"
                    cv2.putText(image, text, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                    # Wait for 's' key to be pressed
                    if cv2.waitKey(10) & 0xFF == ord('s'):
                        print("  ...RECORDING...")
                        break
                
                # --- Start the actual recording ---
                video_name = f'{sequence+1}.mp4'
                video_path = os.path.join(action_path, video_name)
                
                # Define video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

                # Record for 'sequence_length' frames
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    
                    # Process with MediaPipe
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    
                    # Add "Recording..." text
                    cv2.putText(image, f"RECORDING... {action} - Video {sequence+1}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                    # Write the frame to the video file
                    out.write(frame)
                    
                    # Wait a bit (controls frame rate)
                    cv2.waitKey(50)

                out.release()
                print(f"  Saved video {video_path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    # Check if DATA_PATH exists
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        
    collect_data()


