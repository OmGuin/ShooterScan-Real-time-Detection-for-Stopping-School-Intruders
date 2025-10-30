import os
import face_recognition
database_path = "./sample_database"
known_face_encodings = []
known_face_names = []



def load_known_faces():
    for filename in os.listdir(database_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(database_path, filename)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(filename.split('.')[0])
load_known_faces()

if __name__ == "__main__":
    #print(known_face_encodings)
    
    print("Database setup for: ")
    print("\n".join(known_face_names))