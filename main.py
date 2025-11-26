import cv2
import mediapipe as mp
import numpy as np
import winsound
import threading
import time
import csv                  # <--- NEW: For saving data
from datetime import datetime  # <--- NEW: For getting the time

# --- CONFIGURATION ---
THRESH_FRAMES = 20
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.5

# Global variables
COUNTER = 0
ALARM_ON = False
LOG_FILE = "driver_log.csv"  # The name of your Excel file

# --- LOGGING FUNCTION ---


def log_alert(alert_type):
    """Saves the alert to a CSV file"""
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")

    # Open the file in 'append' mode ('a') so we don't delete old data
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([current_time, alert_type])
    print(f"[LOG] Saved {alert_type} event at {current_time}")


# --- MEDIAPIPE SETUP ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True
)

# --- LANDMARK INDICES ---
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308]

# --- HELPER FUNCTIONS ---


def sound_alarm(type="drowsy"):
    if type == "drowsy":
        winsound.Beep(2500, 1000)
    else:
        winsound.Beep(1000, 1000)


def euclidean_distance(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def get_ear(landmarks, indices, img_w, img_h):
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        coord = np.array([int(lm.x * img_w), int(lm.y * img_h)])
        coords.append(coord)
    P2_P6 = euclidean_distance(coords[1], coords[5])
    P3_P5 = euclidean_distance(coords[2], coords[4])
    P1_P4 = euclidean_distance(coords[0], coords[3])
    ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
    return ear, coords


def get_mar(landmarks, indices, img_w, img_h):
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        coord = np.array([int(lm.x * img_w), int(lm.y * img_h)])
        coords.append(coord)
    A = euclidean_distance(coords[0], coords[1])
    B = euclidean_distance(coords[2], coords[3])
    mar = A / B
    return mar, coords


# --- INITIALIZE LOG FILE ---
# Write headers if the file doesn't exist yet
try:
    with open(LOG_FILE, mode='x', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Event Type"])  # Excel Headers
except FileExistsError:
    pass  # File already exists, we will just append to it

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)

print("[INFO] Starting System...")
print(f"[INFO] Logging data to {LOG_FILE}")
print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            left_ear, left_coords = get_ear(landmarks, LEFT_EYE, w, h)
            right_ear, right_coords = get_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            mar, mouth_coords = get_mar(landmarks, MOUTH, w, h)

            cv2.polylines(frame, [np.array(left_coords)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [np.array(right_coords)],
                          True, (0, 255, 0), 1)
            cv2.polylines(frame, [np.array(mouth_coords)],
                          True, (255, 0, 0), 1)

            # --- LOGIC CHECKS ---

            # 1. Yawning
            if mar > MAR_THRESHOLD:
                cv2.putText(frame, "YAWNING DETECTED", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if not ALARM_ON:
                    ALARM_ON = True
                    # LOG THE DATA
                    log_alert("Yawn")
                    t = threading.Thread(target=sound_alarm, args=("yawn",))
                    t.daemon = True
                    t.start()
                ALARM_ON = False

            # 2. Drowsiness
            elif avg_ear < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= THRESH_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        # LOG THE DATA
                        log_alert("Drowsiness")
                        t = threading.Thread(
                            target=sound_alarm, args=("drowsy",))
                        t.daemon = True
                        t.start()

                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                ALARM_ON = False

            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
