import os
import cvzone
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import face_recognition
import requests
from datetime import datetime, timedelta
import time
from collections import deque
from threading import Thread

print("Dependencies Imported")

facemodel = YOLO(r'C:\Users\omgui\Downloads\sciencefair\results\yolov11n-face.pt')
print("Model Loaded")
database_path = r"C:\Users\omgui\Downloads\sciencefair\doorbellcam\sample_database"
known_encodings = []
known_names = []

def open_door():
    #requests.get("http://ompi4.local:5000/on")
    time.sleep(4)

for file in os.listdir(database_path):
    if file.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(database_path, file)
        img = face_recognition.load_image_file(img_path)
        
        encodings = face_recognition.face_encodings(img)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(file)[0])
print(f"Database set up for: {known_names}")



# ---- Try to force FFMPEG capture (often better for HTTP/MJPEG) ----
# (Harmless if not supported on your build)
CAP_BACKEND = cv2.CAP_FFMPEG

camera = "http://ompi3:8080/video_feed_0"

# Reduce internal buffering if supported (has effect mostly on V4L/DirectShow, but cheap to try)
def set_capture_low_latency(cap):
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

# ---- Threaded frame grabber that always keeps only the latest frame ----
class LatestFrameReader:
    def __init__(self, src, backend=None, reconnect=True):
        self.src = src
        self.backend = backend
        self.reconnect = reconnect
        self.cap = None
        self.q = deque(maxlen=1)
        self.running = False
        self.th = None

    def _open(self):
        self.cap = cv2.VideoCapture(self.src, self.backend) if self.backend is not None else cv2.VideoCapture(self.src)
        set_capture_low_latency(self.cap)

    def start(self):
        if self.running:
            return
        self.running = True
        self._open()
        self.th = Thread(target=self._reader, daemon=True)
        self.th.start()

    def _reader(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                if not self.reconnect:
                    break
                # try to reconnect after brief sleep
                time.sleep(0.5)
                self._open()
                continue

            ok, frame = self.cap.read()
            if not ok:
                # brief pause, then try to reopen
                time.sleep(0.1)
                if self.reconnect:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self._open()
                continue
            self.q.append(frame)

    def read_latest(self):
        return self.q[-1] if self.q else None

    def stop(self):
        self.running = False
        if self.th is not None:
            self.th.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()

# ---------------- Your processing loop, optimized ----------------
print("Starting Monitoring Feed...")
reader = LatestFrameReader(camera, backend=CAP_BACKEND, reconnect=True)
reader.start()

frames_detected = 0
prev_time = datetime.now() - timedelta(seconds=20)

# Process at a target rate (e.g., 12 FPS display), but run face work less often
TARGET_FPS = 12
FACE_FPS   = 6     # run face model every ~6 FPS (skip frames in between)
face_interval = max(1, TARGET_FPS // max(1, FACE_FPS))
frame_idx = 0

# Precompute display size once
DISP_W, DISP_H = 700, 500

# Optional: pin OpenCV threads low to reduce jitter from internal parallelism
try:
    cv2.setNumThreads(1)
except Exception:
    pass

last_display = time.time()
frame_period  = 1.0 / TARGET_FPS

while True:
    frame = reader.read_latest()
    if frame is None:
        # no frame yet
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.005)
        continue

    # Downscale first (cheaper overall), stay in BGR for display; only convert crops when needed
    disp = cv2.resize(frame, (DISP_W, DISP_H), interpolation=cv2.INTER_AREA)

    # Run face model only on some frames to keep pipeline real-time
    run_face_now = (frame_idx % face_interval == 0)

    label_overlays = []  # (x1,y1,x2,y2,label)
    if run_face_now:
        # Convert once to RGB for the detector if it requires RGB (many do).
        # If your facemodel expects BGR, remove this conversion.
        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

        # Run detection on the resized frame
        face_result = facemodel.predict(disp_rgb, conf=0.40)

        for info in face_result:
            for box in info.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                # Safe crop with padding from the RGB image (since face_recognition expects RGB)
                pad = 50
                ys = max(0, y1 - pad)
                ye = min(disp_rgb.shape[0], y2 + pad)
                xs = max(0, x1 - pad)
                xe = min(disp_rgb.shape[1], x2 + pad)
                face_img = disp_rgb[ys:ye, xs:xe]

                if face_img.size == 0:
                    continue

                # Downsize crop for encoding cost
                face_img = cv2.resize(face_img, (256, 256), interpolation=cv2.INTER_AREA)

                # Heavy call — do it only on the chosen frames
                encs = face_recognition.face_encodings(face_img)
                label = "Unknown"
                if encs:
                    detected_encoding = encs[0]
                    distances = face_recognition.face_distance(known_encodings, detected_encoding)
                    min_idx = int(np.argmin(distances))
                    if distances[min_idx] < 0.6:
                        status = "Waiting for Verification"
                        name = known_names[min_idx]
                        if frames_detected >= 3:
                            status = "Verified"
                        label = f"{name} {status}"
                        frames_detected += 1

                        # Door logic throttled (left commented as in your code)
                        if frames_detected >= 3 and datetime.now() >= prev_time + timedelta(seconds=20):
                            prev_time = datetime.now()
                            frames_detected = 0
                            open_door()

                label_overlays.append((x1, y1, x2, y2, label))

    # Draw overlays (whether from this frame or last run). For simplicity we draw only when we ran face.
    for (x1, y1, x2, y2, label) in label_overlays:
        w, h = x2 - x1, y2 - y1
        try:
            cvzone.cornerRect(disp, [x1, y1, w, h], l=9, rt=3)
        except Exception:
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(disp, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Display at target FPS (simple scheduler)
    now = time.time()
    delay = frame_period - (now - last_display)
    if delay > 0:
        time.sleep(delay)
    last_display = time.time()

    cv2.imshow('frame', disp)
    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

reader.stop()
cv2.destroyAllWindows()
