import threading 
import cv2
from deepface import DeepFace
import os
import postgres_connection as pc

cursor,conn = pc.connect_db()

models = [
    "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",
    "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet",
    "Buffalo_L",
]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


script_dir = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(os.path.join(script_dir, "images", "foto1.jpg"))

counter = 0

face_matches = {}

def checkFace(face_roi, face_id):
    global face_matches
    try:
        result = DeepFace.verify(face_roi, img.copy(), enforce_detection=False)
        face_matches[face_id] = result['verified']
    except Exception:
        face_matches[face_id] = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        faces_data = DeepFace.extract_faces(frame, enforce_detection=False)
    except Exception:
        faces_data = []
    
    if counter % 30 == 0:
        for face_data in faces_data:
            face_roi = face_data['face']
            facial_area = face_data['facial_area']
            face_id = f"{facial_area['x']}_{facial_area['y']}_{facial_area['w']}_{facial_area['h']}"
            try:
                threading.Thread(target=checkFace, args=(face_roi, face_id)).start()
            except Exception:
                pass
    
    for face_data in faces_data:
        facial_area = face_data['facial_area']
        x = facial_area['x']
        y = facial_area['y']
        w = facial_area['w']
        h = facial_area['h']
        
        face_id = f"{x}_{y}_{w}_{h}"
        is_match = face_matches.get(face_id, False)
        
        if is_match:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Match", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Camera", frame)
    
    counter += 1
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()