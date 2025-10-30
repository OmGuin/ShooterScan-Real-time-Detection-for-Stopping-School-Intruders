import os
import cvzone
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace

facemodel = YOLO(r'C:\Users\omgui\Downloads\sciencefair\results\yolov11n-face.pt', verbose = False)
clips = r"C:\Users\omgui\Downloads\sciencefair\results\clips"


for clip in os.listdir(clips):
    video_path = os.path.join(clips, clip)
    cap = cv2.VideoCapture(video_path)
    print(f"Processing {clip}...")


    while True:
        rt, frame = cap.read()
        if not rt:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
    
        rgb_frame = frame[:, :, ::-1]  # Convert to RGB
        video_resized = cv2.resize(rgb_frame, (360, 640))

        
        face_result = facemodel.predict(video_resized, conf=0.40)
        cv2.putText(
            video_resized, 
            clip, 
            (10, 30),  # Position of the text (x, y)
            cv2.FONT_HERSHEY_SIMPLEX, 
            1,  # Font size
            (255, 255, 255),  # Text color (white)
            2,  # Line thickness
            cv2.LINE_AA
        )


        for info in face_result:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h, w = y2 - y1, x2 - x1
                label = "unknown"
                face_img = video_resized[max(0, y1-50):min(y2+50, video_resized.shape[0]), max(0, x1-50):min(x2+50, video_resized.shape[1])]
                analysis = DeepFace.analyze(img_path = face_img, actions=['emotion'], enforce_detection = False)
                dominant_emotion = analysis[0]['dominant_emotion']
                
                label = dominant_emotion
                cvzone.cornerRect(video_resized, [x1, y1, w, h], l=9, rt=3)
                cv2.putText(video_resized, label , (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        video_resized = video_resized[:, :, ::-1]
        cv2.imshow('frame', video_resized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    cap.release()
cv2.destroyAllWindows()
