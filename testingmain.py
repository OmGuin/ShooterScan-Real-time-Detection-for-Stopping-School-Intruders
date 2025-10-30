import os
import cvzone
from ultralytics import YOLO
import cv2
import numpy as np
import face_recognition
import requests
from datetime import datetime, timedelta
import time
from collections import deque
from threading import Thread, Lock


facemodel = YOLO(r'C:\Users\omgui\Desktop\SS\results\yolov11n-face.pt')
database_path = r"C:\Users\omgui\Desktop\SS\doorbellcam\sample_database"
known_encodings = []
known_names = []

def open_door():
    requests.get("http://ompi4.local:5000/on")

class DoorOpener:
    """Runs open_door() on a background thread with a cooldown and single-flight guard."""
    def __init__(self, cooldown_sec: float = 20.0):
        self.cooldown = cooldown_sec
        self._lock = Lock()
        self._running = False
        self._last_ts = 0.0

    def trigger(self) -> bool:
        """Try to start a door-open operation in the background.
        Returns True if it actually started; False if blocked by cooldown/already running.
        """
        now = time.time()
        with self._lock:
            if self._running:
                return False
            if now - self._last_ts < self.cooldown:
                return False
            self._running = True
            self._last_ts = now
        Thread(target=self._worker, daemon=True).start()
        return True

    def _worker(self):
        try:
            open_door()
            print("[DoorOpener] Door action completed.")
        except Exception as e:
            print(f"[DoorOpener] Error: {e}")
        finally:
            with self._lock:
                self._running = False

door_opener = DoorOpener(cooldown_sec=20.0)
# ----------------------------------------------------------

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
CAP_BACKEND = cv2.CAP_FFMPEG
camera = "http://ompi3:8080/video_feed_0"

def set_capture_low_latency(cap):
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

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
                time.sleep(0.5)
                self._open()
                continue
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.1)
                if self.reconnect:
                    try: self.cap.release()
                    except Exception: pass
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

print("Starting Monitoring Feed...")
reader = LatestFrameReader(camera, backend=CAP_BACKEND, reconnect=True)
reader.start()

frames_detected = 0

TARGET_FPS = 10
FACE_FPS   = 10
face_interval = max(1, TARGET_FPS // max(1, FACE_FPS))
frame_idx = 0

DISP_W, DISP_H = 700, 500

try:
    cv2.setNumThreads(1)
except Exception:
    pass

last_display = time.time()
frame_period  = 1.0 / TARGET_FPS

while True:
    frame = reader.read_latest()
    if frame is None:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.005)
        continue

    disp = cv2.resize(frame, (DISP_W, DISP_H), interpolation=cv2.INTER_AREA)

    run_face_now = (frame_idx % face_interval == 0)
    label_overlays = []
    if run_face_now:
        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        face_result = facemodel.predict(disp_rgb, conf=0.40, verbose=False)

        for info in face_result:
            for box in info.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                pad = 50
                ys = max(0, y1 - pad)
                ye = min(disp_rgb.shape[0], y2 + pad)
                xs = max(0, x1 - pad)
                xe = min(disp_rgb.shape[1], x2 + pad)
                face_img = disp_rgb[ys:ye, xs:xe]
                if face_img.size == 0:
                    continue

                face_img = cv2.resize(face_img, (256, 256), interpolation=cv2.INTER_AREA)
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
                            # >>> Non-blocking trigger; returns False if on cooldown or already running
                            if door_opener.trigger():
                                print("[DoorOpener] Triggered.")
                            # reset the verification counter after a successful trigger
                            frames_detected = 0
                        else:
                            frames_detected += 1
                        label = f"{name} {status}"

                label_overlays.append((x1, y1, x2, y2, label))

    for (x1, y1, x2, y2, label) in label_overlays:
        w, h = x2 - x1, y2 - y1
        try:
            cvzone.cornerRect(disp, [x1, y1, w, h], l=9, rt=3)
        except Exception:
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(disp, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

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
