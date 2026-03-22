import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import TensorBoard

# SETTINGS
KEYPOINT_PATH = 'set your path here'  # Path to optimized keypoint data
SEQUENCE_LENGTH = 60 # Frames per video
MODEL_NAME = 'set your model name here' # Name for the saved model

# 1. Load Data (Same as before)
print("Loading data...")
actions = []
sequences = []
labels = []

# Load data from folders
for action in os.listdir(KEYPOINT_PATH):
    action_path = os.path.join(KEYPOINT_PATH, action)
    if not os.path.isdir(action_path):
        continue
    actions.append(action)
    for sequence_file in os.listdir(action_path):
        if sequence_file.endswith('.npy'):
            res = np.load(os.path.join(action_path, sequence_file))
            sequences.append(res)
            labels.append(len(actions) - 1)

print(f"Found {len(sequences)} sequences for {len(actions)} actions: {actions}")

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

# 2. Build the CNN Model
# We use Conv1D because we are processing a sequence of numbers (time)

model = Sequential()

# Layer 1: Look for small patterns (3 frames long)
#understand small patterns in the sequence
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(SEQUENCE_LENGTH, 126)))# relu(retified linear unit) activation function
model.add(BatchNormalization()) 
model.add(MaxPooling1D(pool_size=2)) # Reduces data size and keeps important features!!

# Layer 2: Look for larger patterns
#understand combined movemnents
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# Layer 3: High level features
# regonize full sequences
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())

# Flatten: Turn the sequence into a single list of features , 3d into 1D
model.add(Flatten())

# Classification Layers
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5)) # Prevents overfitting
model.add(Dense(len(actions), activation='softmax'))

# Compile
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# 3. Train
print("\n--- Starting CNN Training ---")
model.fit(X_train, y_train, 
          epochs=100, 
          batch_size=32,
          validation_data=(X_test, y_test))

# 4. Save Model
model.save(MODEL_NAME)
print(f"CNN Model saved as {MODEL_NAME}")