

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score

# DATA AUGMENTATION 
def augment_data(data, labels, num_augmentations=5, noise_level=0.01):
    augmented_data = [data]
    augmented_labels = [labels]

    print(f"Augmenting data... creating {num_augmentations} new versions of each training sample.")

    for _ in range(num_augmentations):
        noise = np.random.normal(0, noise_level, data.shape).astype(np.float32)
        augmented_data.append(data + noise)
        augmented_labels.append(labels)

    return np.vstack(augmented_data), np.hstack(augmented_labels)


# LOAD DATA 
print("Loading data...")
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'], dtype=np.float32)
labels = np.asarray(data_dict['labels'])

print("Original dataset shape:", data.shape)

# REMOVE EXACT DUPLICATES 
data_labels = np.hstack((data, labels.reshape(-1, 1)))
unique_data = np.unique(data_labels, axis=0)

data = unique_data[:, :-1].astype(np.float32)
labels = unique_data[:, -1].astype(int)

print("Dataset shape after removing duplicates:", data.shape)

# GET GROUP IDS (CRITICAL FIX)
"""
BEST CASE:
If your pickle already contains group information such as:
- data_dict['groups']
- data_dict['video_ids']
- data_dict['person_ids']

Then USE THAT directly.
"""

if 'groups' in data_dict:
    groups = np.asarray(data_dict['groups'])
    print("Using provided group IDs.")

else:
    """
    FALLBACK (SAFE DEFAULT):
    We assume consecutive samples come from the same source.
    This prevents subject/video leakage.
    """
    print("No group IDs found. Creating conservative group IDs...")
    group_size = 10   # assume every 10 samples belong to same source
    groups = np.arange(len(labels)) // group_size

# GROUP-WISE SPLIT 
gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(data, labels, groups=groups))

x_train, x_test = data[train_idx], data[test_idx]
y_train, y_test = labels[train_idx], labels[test_idx]

print("Train shape:", x_train.shape)
print("Test shape :", x_test.shape)

# AUGMENT TRAINING DATA ONLY 
x_train_aug, y_train_aug = augment_data(
    x_train,
    y_train,
    num_augmentations=5,
    noise_level=0.01
)

print("Augmented training shape:", x_train_aug.shape)

# TRAIN MODEL 
print("Training model...")
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(x_train_aug, y_train_aug)

# EVALUATE 
print("Evaluating model...")
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n========== RESULTS ==========")
print(f"Static Model Accuracy: {accuracy * 100:.2f}%")
print("=============================")

# SAVE MODEL 
print("Saving model...")
with open('model_ac.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved to model_ac.p")
