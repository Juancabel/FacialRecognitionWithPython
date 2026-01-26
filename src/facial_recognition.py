import threading 
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

img = cv2.imread("images/juan_camilo.jpeg")

counter = 0

face_match = False

def checkFace():

    global face_match
    try:
        result = DeepFace.verify(frame, img.copy(), enforce_detection=False)
        face_match = result['verified']
    except ValueError:
        pass

while True:
    ret, frame = cap.read()
    if ret:
        if counter % 20 == 0:
            try: 
                threading.Thread(target = checkFace, args = (frame.copy(),)).start()
            except ValueError:
                pass
        if face_match:
            cv2.putText(frame, "Face Matched", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    counter += 1
    key = cv2.waitKey(1)
    if key == ord('q'):
        break






cv2.destroyAllWindows()