# 🤟 Malayalam Sign Language Learning Platform

An AI-powered system designed to recognize and assist in learning Malayalam Sign Language using Computer Vision, Machine Learning, and Deep Learning.

---

## 🚀 Project Overview

This project aims to bridge the communication gap by enabling real-time recognition of Malayalam sign language. The system captures hand gestures using a webcam and converts them into Malayalam text and speech output.

The project is divided into two phases:

- **Phase 1:** Static gesture recognition (letters)
- **Phase 2:** Dynamic gesture recognition (words)

---

## ✨ Key Features

- 🎥 Real-time gesture recognition using webcam
- ✋ Accurate hand landmark detection using MediaPipe
- 🔤 Supports Malayalam sign language
- 🧠 Uses Machine Learning and Deep Learning models
- 🔊 Converts gestures into:
  - Malayalam text
  - Speech output (Text-to-Speech)
- ⚡ Fast and efficient real-time processing

---

## 🧠 Technologies Used

| Category             | Tools / Libraries            |
| -------------------- | ---------------------------- |
| Programming Language | Python                       |
| Computer Vision      | OpenCV, MediaPipe            |
| Machine Learning     | Scikit-learn (Random Forest) |
| Deep Learning        | TensorFlow / Keras (CNN)     |
| Data Handling        | NumPy                        |
| Model Storage        | Pickle (.p), HDF5 (.h5)      |

---

## 🏗️ System Architecture

The system follows a structured pipeline:

1. Webcam captures input
2. MediaPipe extracts hand landmarks
3. Data is processed and routed
4. Model predicts gesture
   - Random Forest → Static gestures
   - CNN → Dynamic gestures
5. Output is displayed as text and speech

---

## 🔄 Workflow

### 🔹 Phase 1 – Static Gesture Recognition

- Input: Image (single frame)
- Feature Extraction: Hand landmarks (x, y)
- Model: Random Forest
- Output: Malayalam letter

### 🔹 Phase 2 – Dynamic Gesture Recognition

- Input: Video sequence
- Feature Extraction: Keypoints (x, y, z)
- Sequence Shape: (60 × 126)
- Model: 1D CNN
- Output: Malayalam word

---

## 📊 Performance

| Model Type              | Accuracy |
| ----------------------- | -------- |
| Static Model            | 99–100%  |
| Dynamic Model (Letters) | ~98%     |
| Dynamic Model (Words)   | ~93%     |

---

## 📁 Project Structure
