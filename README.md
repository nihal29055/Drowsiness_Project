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
```
---

## 2. Install Dependencies
   Run the following command to install the required libraries:
```pip install opencv-python mediapipe numpy```
💻 Usage
Make sure your webcam is connected.
Run the main script:
```python main.py```

---

## 3.Controls:

The window titled "Driver Monitoring System" will appear.
Press `q` on your keyboard to quit the program.

---

## 🧠 How It Works
1. Face Mesh: The system uses MediaPipe to map 468 facial landmarks.
2. EAR (Eye Aspect Ratio): We extract 6 coordinates for each eye. The ratio of vertical distance to horizontal distance determines if the eye is open or closed.
3. MAR (Mouth Aspect Ratio): We extract top/bottom and left/right lip coordinates. A high vertical ratio indicates a yawn.
4. Logic:
~ If `MAR > Threshold` → Yawn Alert
~ If `EAR < Threshold` for `X` frames → Drowsiness Alert

---

## 🔧 Troubleshooting

* "ModuleNotFoundError": Ensure you activated your virtual environment (`.\venv\Scripts\Activate.ps1`) before running the script.
* Webcam not opening: Change the index in the code `cv2.VideoCapture(0)` to `1 `or `2` if you have multiple cameras.
* Audio not playing: This project uses` winsound` (Windows only). If you are on Mac/Linux, replace the `sound_alarm` function with a cross-platform library like` playsound`.

---

## 🔮 Future Scope

* Data Logging: Save a CSV file with timestamps of every alert for driver analysis.
* Mobile App: Convert the model to TFLite for use on Android/iOS.
* Head Pose Estimation: Add detection for when the driver looks away from the road (distraction).
---

## 📄 License
This project is open-source and available under the MIT License.
