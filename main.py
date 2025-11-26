import cv2
import mediapipe as mp
import numpy as np
import winsound
import threading
import time

# --- CONFIGURATION ---
THRESH_FRAMES = 20      # Frames eyes must be closed to trigger alarm
EAR_THRESHOLD = 0.25    # Eye Aspect Ratio (Eyes closed)
MAR_THRESHOLD = 0.5     # Mouth Aspect Ratio (Mouth open/Yawning)

# Global variables
COUNTER = 0
ALARM_ON = False

# --- MEDIAPIPE SETUP ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True
)

# --- LANDMARK INDICES ---
# Eyes
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Mouth (Top, Bottom, Left, Right)
MOUTH = [13, 14, 78, 308]

# --- HELPER FUNCTIONS ---


def sound_alarm(type="drowsy"):
    """
    Plays different sounds for different alerts.
    Drowsy = High Pitch Beep
    Yawn = Low Pitch Beep
    """
    if type == "drowsy":
        winsound.Beep(2500, 1000)  # High pitch
    else:
        winsound.Beep(1000, 1000)  # Low pitch


def euclidean_distance(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def get_ear(landmarks, indices, img_w, img_h):
    """Calculates Eye Aspect Ratio"""
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
    """Calculates Mouth Aspect Ratio"""
    # 0=Top, 1=Bottom, 2=Left, 3=Right
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        coord = np.array([int(lm.x * img_w), int(lm.y * img_h)])
        coords.append(coord)

    # Vertical Distance (Top lip to Bottom lip)
    A = euclidean_distance(coords[0], coords[1])
    # Horizontal Distance (Left corner to Right corner)
    B = euclidean_distance(coords[2], coords[3])

    mar = A / B
    return mar, coords


# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)

print("[INFO] Starting System...")
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

            # 1. Calculate EAR (Eyes)
            left_ear, left_coords = get_ear(landmarks, LEFT_EYE, w, h)
            right_ear, right_coords = get_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            # 2. Calculate MAR (Mouth)
            mar, mouth_coords = get_mar(landmarks, MOUTH, w, h)

            # 3. Draw Visuals
            cv2.polylines(frame, [np.array(left_coords)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [np.array(right_coords)],
                          True, (0, 255, 0), 1)
            cv2.polylines(frame, [np.array(mouth_coords)],
                          True, (255, 0, 0), 1)

            # --- LOGIC CHECKS ---

            # Check 1: Yawning (Mouth Open)
            if mar > MAR_THRESHOLD:
                cv2.putText(frame, "YAWNING DETECTED", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if not ALARM_ON:
                    ALARM_ON = True
                    t = threading.Thread(target=sound_alarm, args=("yawn",))
                    t.daemon = True
                    t.start()
                ALARM_ON = True

            # Check 2: Drowsiness (Eyes Closed)
            elif avg_ear < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= THRESH_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        t = threading.Thread(
                            target=sound_alarm, args=("drowsy",))
                        t.daemon = True
                        t.start()

                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                ALARM_ON = False

            # Display Values on Screen
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
