# Driver Drowsiness & Yawning Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

A real-time safety system that monitors driver fatigue using Computer Vision. This project uses **MediaPipe** for facial landmark detection and **OpenCV** for image processing to detect eye closure (drowsiness) and yawning, triggering an audible alarm to alert the driver.

---

## 🚀 Features

* **Real-time Detection:** Uses MediaPipe Face Mesh for high-speed, accurate landmark tracking (no heavy Dlib installation required).
* **Drowsiness Detection:** Calculates **Eye Aspect Ratio (EAR)** to detect prolonged eye closure.
* **Yawn Detection:** Calculates **Mouth Aspect Ratio (MAR)** to detect yawning events.
* **Visual Feedback:** Draws contours around eyes and mouth with live EAR/MAR data values on screen.
* **Audio Alerts:** distinct alarms for "Drowsiness" (High pitch) and "Yawning" (Low pitch) using Windows native sound.

---

## 🛠️ Tech Stack & Requirements

* **OS:** Windows (Tested). *Note: The audio alarm uses `winsound`, which is built into Windows. For Mac/Linux, you will need to swap this for `playsound` or `pydub`.*
* **Python:** Version 3.8 – 3.12 recommended.
* **Hardware:** Standard Webcam.

### Python Libraries
* `opencv-python` (Video processing)
* `mediapipe` (Facial landmarks)
* `numpy` (Math calculations)

---

## 📦 Installation Guide

### 1. Set up the Environment (PowerShell)
It is recommended to use a virtual environment to keep your project clean. Open PowerShell in your project folder:

```powershell
# Create virtual environment
python -m venv venv

# Activate environment
.\venv\Scripts\Activate.ps1

# Upgrade pip (optional but recommended)
python -m pip install --upgrade pip setuptools wheel