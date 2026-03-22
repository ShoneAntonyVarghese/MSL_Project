

import os
import cv2

# SETTINGS 
DATA_DIR = 'set your data directory here' # <-- CHANGE THIS
number_of_classes = 10
dataset_size = 700 # Number of images to collect per class

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Use camera index
cap = cv2.VideoCapture(1) 
if not cap.isOpened():
    print("Error: Could not open camera. Try changing the index in cv2.VideoCapture().")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Prompt user to get ready
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Flip 
        frame = cv2.flip(frame, 1)

        # Add text to the frame
        cv2.putText(frame, 'Ready? Press "Q" to start!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Wait for 'q' key to be pressed
        if cv2.waitKey(25) == ord('Q'):
            break

    # Start capturing data
    print('Capturing...')
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame during data collection.")
            break

        # Flip the frame consistently
        frame = cv2.flip(frame, 1)
        
        # Show the frame being captured
        cv2.imshow('frame', frame)
        cv2.waitKey(25) # Add a small delay

        # Save the frame to the directory
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()