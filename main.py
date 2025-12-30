import cv2
import os
import time
from playsound import playsound
import threading

# ================= Setup =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Haar cascades
eye_cascade_path = os.path.join(BASE_DIR, "haarcascade_eye.xml")
car_cascade_path = os.path.join(BASE_DIR, "haarcascade_car.xml")

eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
car_cascade = cv2.CascadeClassifier(car_cascade_path)

if eye_cascade.empty() or car_cascade.empty():
    print("Error: Could not load cascade files.")
    exit()

# ================= Parameters =================
ALARM_TIME = 3  # seconds before alarm triggers
eye_closed_start = None
alarm_played = False  # Track if alarm has played

# Alarm MP3 path (raw string for Windows)
ALARM_SOUND = r"C:\Users\DELL\Downloads\preview.mp3"

# Initialize status variables
status = "Initializing..."
color = (0, 255, 0)

# Function to play alarm without blocking the main thread
def play_alarm():
    playsound(ALARM_SOUND)

# ================= Camera Capture =================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------- Detect eyes --------
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # -------- Eye Closure Logic --------
    if len(eyes) > 0:
        # Eyes detected → reset timer and alarm
        eye_closed_start = None
        alarm_played = False
        status = "Eyes Open"
        color = (0, 255, 0)
    else:
        # Eyes NOT detected
        if eye_closed_start is None:
            eye_closed_start = time.time()
        elif time.time() - eye_closed_start >= ALARM_TIME:
            status = "ALERT! Eyes Closed"
            color = (0, 0, 255)
            if not alarm_played:
                # Play alarm in a separate thread
                threading.Thread(target=play_alarm, daemon=True).start()
                alarm_played = True
        else:
            status = "Eyes Closed"
            color = (0, 255, 255)

    cv2.putText(frame, status, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Draw rectangles around eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # -------- Detect cars --------
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # -------- Show frame --------
    cv2.imshow("Detection & Eye Alarm", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
