import cvzone
from ultralytics import YOLO
import cv2
import os 
import sys
import torch
import torch.nn as nn
from torchvision import transforms
#sys.stdout = open(os.devnull, 'w')
import numpy as np
from PIL import Image
#from fer import FER
#from deepface import DeepFace
import face_recognition
from databasesetup import known_face_encodings, known_face_names
import time

video = 0
cap = cv2.VideoCapture(video)

#----------------------------------- YOLO 11 METHOD
facemodel = YOLO('yolov11n-face.pt')

database_path = 'C:../sample_database"

#print(known_face_encodings)
print(known_face_names)



while True:
    rt, frame = cap.read()
    rgb_frame = frame[:, :, ::-1]
    rgb_frame = cv2.resize(rgb_frame, (256, 256))
    face_locations = face_recognition.face_locations(rgb_frame, model = "HOG")
    
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"  

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        

    time.sleep(0.1)
    cv2.imshow('Real-time Face Recognition', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

