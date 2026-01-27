import threading 
import cv2
from deepface import DeepFace
import os
import postgres_connection as pc

cursor,conn = pc.connect_db()

#pc.reset_table(cursor, conn)
#pc.insert_images_from_file(cursor,conn,os.path.join(os.path.dirname(os.path.abspath(__file__)), "images/"))


models = [
    "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",
    "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet",
    "Buffalo_L",
]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


script_dir = os.path.dirname(os.path.abspath(__file__))

counter = 0

last_results = {}
face_analysis = {}  
lock = threading.Lock()

def get_closest_stored_face(x, y, w, h, threshold=50):
    """Find a stored face result that's close to the given coordinates"""
    closest = None
    min_distance = threshold
    
    for stored_id, result in last_results.items():
        parts = stored_id.split('_')
        if len(parts) != 4:
            continue
        sx, sy, sw, sh = map(int, parts)
        distance = ((x - sx)**2 + (y - sy)**2)**0.5
        if distance < min_distance:
            min_distance = distance
            closest = result
    
    return closest if closest else {"name": None, "matched": False}

def get_closest_analysis(x, y, w, h, threshold=50):
    closest = None
    min_distance = threshold
    
    for stored_id, analysis in face_analysis.items():
        parts = stored_id.split('_')
        if len(parts) != 4:
            continue
        sx, sy, sw, sh = map(int, parts)
        distance = ((x - sx)**2 + (y - sy)**2)**0.5
        if distance < min_distance:
            min_distance = distance
            closest = analysis
    
    return closest if closest else {"emotion": "", "age": 0}

def checkFace(face_roi, face_id):
    global last_results
    try:
        target = pc.get_representation(face_roi)
        result = pc.search_similar_faces(cursor, conn, target)
        with lock:
            last_results[face_id] = result
    except Exception as e:
        with lock:
            last_results[face_id] = {
                "name": None,
                "avg_distance": None,
                "matched": False
            }
        print("There was an exception processing the face:", e)

def analyzeFace(face_roi, face_id):

    global face_analysis
    try:
        
        analysis = DeepFace.analyze(face_roi, actions=['emotion', 'age'], enforce_detection=False)
        
        with lock:
            face_analysis[face_id] = {
                "emotion": analysis[0]['dominant_emotion'],
                "age": int(analysis[0]['age'])
            }
    except Exception as e:
        print(f"Error analyzing face {face_id}: {e}")
        with lock:
            face_analysis[face_id] = {"emotion": "Unknown", "age": 0}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        faces_data = DeepFace.extract_faces(frame, enforce_detection=False)
    except Exception:
        faces_data = []
    
    if counter % 60 == 0:
        for face_data in faces_data:
            face_roi = face_data['face']
            facial_area = face_data['facial_area']
            face_id = f"{facial_area['x']}_{facial_area['y']}_{facial_area['w']}_{facial_area['h']}"
            if face_roi.max() <= 1.0:
                face_roi = (face_roi * 255).astype('uint8')
            try:
                threading.Thread(target=checkFace, args=(face_roi, face_id)).start()
            except Exception:
                pass
    
    if counter % 120 == 0:
        for face_data in faces_data:
            facial_roi = face_data['face']
            face_id = f"{facial_area['x']}_{facial_area['y']}_{facial_area['w']}_{facial_area['h']}"
            try:
                threading.Thread(target=analyzeFace, args=(face_roi, face_id)).start()
            except Exception:
                pass
    
    
    for face_data in faces_data:
        facial_area = face_data['facial_area']
        x = facial_area['x']
        y = facial_area['y']
        w = facial_area['w']
        h = facial_area['h']
        
        face_id = f"{x}_{y}_{w}_{h}"
        with lock:
            result = get_closest_stored_face(x, y, w, h)
            analysis = get_closest_analysis(x, y, w, h)
        
        text_lines = []
        if isinstance(result, dict) and result.get("matched"):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            name = result.get("name", "Unknown")
            distance = result.get("avg_distance", 0)
            text_lines.append(f"{name} ({distance:.2f})")
            color = (0, 255, 0)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            text_lines.append("Unknown")
            color = (0, 0, 255)
        
        if analysis.get("emotion"):
            text_lines.append(f"{analysis['emotion']} (age: {analysis['age']})")
        
        for i, text in enumerate(text_lines):
            cv2.putText(frame, text, (x, y - 10 - (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Camera", frame)
    
    counter += 1
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()