import threading 
import cv2
from deepface import DeepFace
import os
import time
from enum import Enum
import postgres_connection as pc

#THINGS TO KEEP WORKIN ON: 

#ADD CHECKS ON ACTUAL CAPTURE, FOR NOW 20 EMBEDDINGS IS FINE 
#MULTIPLE FACES IN FRAME, FOR NOW ONLY FIRST FACE IS PROCESSED, CAN ADD MORE LATER
#ADD OPTION TO DELETE INDIVIDUAL FACES, FOR NOW ONLY FULL DATABASE CLEAN, CAN ADD LATER


# ============================================================================
# CONSTANTS
# ============================================================================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FACE_RECOGNITION_INTERVAL = 30  # Process face recognition every N frames
EMOTION_RECOGNITION_INTERVAL = 60  # Process emotion recognition every N frames
ADD_FACE_CAPTURE_INTERVAL = 15  # Capture embedding every N frames during add face
ADD_FACE_DURATION_FRAMES = 300  # Total frames for add face capture (~20 captures)

# Colors (BGR format)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_CYAN = (255, 255, 0)


# ============================================================================
# APPLICATION STATE
# ============================================================================

class AppState(Enum):
    """Enumeration of possible application states."""
    IDLE = 0  # Default state - only face detection with blue rectangles
    FACE_RECOGNITION = 1  # Active face recognition mode
    EMOTION_RECOGNITION = 2  # Active emotion recognition mode
    ADD_FACE_NAME = 3  # Typing name for new face
    ADD_FACE_CAPTURE = 4  # Capturing face embeddings


# Global state variables
current_state = AppState.IDLE
frame_counter = 0
lock = threading.Lock()

# Storage for recognition results
face_recognition_results = {}  # Stores face recognition results {face_id: result}
emotion_recognition_results = {}  # Stores emotion results {face_id: analysis}

# Thread management - prevent launching multiple concurrent threads
face_recognition_in_progress = False  # Flag to track if face recognition thread is running
emotion_recognition_in_progress = False  # Flag to track if emotion recognition thread is running
thread_lock = threading.Lock()  # Lock for thread management flags

# Add face feature variables
add_face_name = ""  # Name being typed
add_face_start_frame = 0  # Frame when capture started
add_face_capture_count = 0  # Number of embeddings captured


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

cursor, conn = pc.connect_db()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_closest_result(x, y, w, h, results_dict, threshold=50):
    """
    Find a stored result that's close to the given face coordinates.
    
    Since faces move slightly between frames, this function finds the closest
    stored result based on Euclidean distance between face positions.
    
    Args:
        x, y, w, h: Current face bounding box coordinates
        results_dict: Dictionary of stored results keyed by face_id
        threshold: Maximum distance to consider a match
    
    Returns:
        The closest matching result or None
    """
    closest = None
    min_distance = threshold
    
    for stored_id, result in results_dict.items():
        parts = stored_id.split('_')
        if len(parts) != 4:
            continue
        sx, sy, sw, sh = map(int, parts)
        distance = ((x - sx)**2 + (y - sy)**2)**0.5
        if distance < min_distance:
            min_distance = distance
            closest = result
    
    return closest


def draw_text(frame, text, position, font_scale=0.6, text_color=COLOR_BLACK):
    """
    Draw text on the frame with a nicer font.
    
    Args:
        frame: The image frame to draw on
        text: Text string to display
        position: (x, y) tuple for text position
        font_scale: Size of the font
        text_color: Color of the text (BGR)
    """
    font = cv2.FONT_HERSHEY_DUPLEX  # Nicer looking font
    thickness = 1
    cv2.putText(frame, text, position, font, font_scale, text_color, thickness)


def draw_instructions(frame, state):
    """
    Draw instruction overlay on the frame based on current state.
    
    Args:
        frame: The image frame to draw on
        state: Current application state
    """
    y_offset = 30
    line_height = 25
    
    if state == AppState.IDLE:
        # Show main menu
        instructions = [
            "=== FACE RECOGNITION SYSTEM ===",
            "[1] Start Face Recognition",
            "[2] Start Emotion Recognition",
            "[3] Add New Face",
            "[4] Clean Database",
            "[Q] Quit"
        ]
        for i, text in enumerate(instructions):
            draw_text(frame, text, (10, y_offset + i * line_height))
    
    elif state == AppState.FACE_RECOGNITION:
        instructions = [
            "=== FACE RECOGNITION ACTIVE ===",
            "Green = Known face | Red = Unknown",
            "[1] Stop Face Recognition"
        ]
        for i, text in enumerate(instructions):
            color = COLOR_GREEN if i == 0 else COLOR_BLACK
            draw_text(frame, text, (10, y_offset + i * line_height), text_color=color)
    
    elif state == AppState.EMOTION_RECOGNITION:
        instructions = [
            "=== EMOTION RECOGNITION ACTIVE ===",
            "Detecting emotions on faces",
            "[2] Stop Emotion Recognition"
        ]
        for i, text in enumerate(instructions):
            color = COLOR_CYAN if i == 0 else COLOR_BLACK
            draw_text(frame, text, (10, y_offset + i * line_height), text_color=color)
    
    elif state == AppState.ADD_FACE_NAME:
        instructions = [
            "=== ADD NEW FACE ===",
            f"Type name: {add_face_name}_",
            "[ENTER] Confirm name",
            "[ESC] Cancel"
        ]
        for i, text in enumerate(instructions):
            color = COLOR_YELLOW if i == 0 else COLOR_BLACK
            draw_text(frame, text, (10, y_offset + i * line_height), text_color=color)
    
    elif state == AppState.ADD_FACE_CAPTURE:
        # Calculate progress based on captures
        elapsed_frames = frame_counter - add_face_start_frame
        progress = min(100, (elapsed_frames / ADD_FACE_DURATION_FRAMES) * 100)
        
        instructions = [
            "=== CAPTURING FACE ===",
            f"Name: {add_face_name}",
            f"Captures: {add_face_capture_count}",
            f"Progress: {progress:.0f}%"
        ]
        for i, text in enumerate(instructions):
            color = COLOR_YELLOW if i == 0 else COLOR_BLACK
            draw_text(frame, text, (10, y_offset + i * line_height), text_color=color)
        
        # Draw progress bar
        bar_width = 200
        bar_height = 20
        bar_x = 10
        bar_y = y_offset + len(instructions) * line_height + 10
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                      COLOR_WHITE, 2)
        fill_width = int((progress / 100) * bar_width)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                      COLOR_GREEN, -1)


def draw_face_rectangle(frame, x, y, w, h, color, label_lines=None):
    """
    Draw a rectangle around a detected face with optional labels.
    
    Args:
        frame: The image frame to draw on
        x, y, w, h: Bounding box coordinates
        color: Rectangle color (BGR)
        label_lines: List of text strings to display above the face
    """
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    if label_lines:
        for i, text in enumerate(label_lines):
            text_y = y - 10 - (i * 25)
            if text_y > 20:  # Only draw if there's space
                draw_text(frame, text, (x, text_y), text_color=color)


# ============================================================================
# FEATURE FUNCTIONS - FACE RECOGNITION
# ============================================================================

def process_face_recognition(face_roi, face_id):
    """
    Process face recognition in a separate thread.
    
    Queries the database to find matching faces and stores the result.
    Sets flag when starting and clears it when done to prevent concurrent threads.
    
    Args:
        face_roi: Cropped face image
        face_id: Unique identifier for this face position
    """
    global face_recognition_results, face_recognition_in_progress
    try:
        target = pc.get_representation(face_roi)
        result = pc.search_similar_faces(cursor, conn, target)
        with lock:
            face_recognition_results[face_id] = result
    except Exception as e:
        with lock:
            face_recognition_results[face_id] = {
                "name": None,
                "avg_distance": None,
                "matched": False
            }
        print(f"Face recognition error: {e}")
    finally:
        # Always clear the in-progress flag when done
        with thread_lock:
            face_recognition_in_progress = False


def handle_face_recognition(frame, faces_data):
    """
    Handle face recognition mode - detect and identify faces.
    
    Only launches a new recognition thread if no other is currently running.
    This prevents CPU/RAM overload from too many concurrent threads.
    
    Args:
        frame: Current video frame
        faces_data: List of detected faces from DeepFace
    """
    global frame_counter, face_recognition_in_progress
    
    # Process recognition every N frames, but only if no thread is already running
    if frame_counter % FACE_RECOGNITION_INTERVAL == 0 and faces_data:
        with thread_lock:
            if face_recognition_in_progress:
                # Skip this frame - previous recognition still in progress
                pass
            else:
                # Mark as in progress and launch thread for first face only
                face_recognition_in_progress = True
                face_data = faces_data[0]  # Process one face at a time
                face_roi = face_data['face']
                facial_area = face_data['facial_area']
                face_id = f"{facial_area['x']}_{facial_area['y']}_{facial_area['w']}_{facial_area['h']}"
                
                # Normalize face image if needed
                if face_roi.max() <= 1.0:
                    face_roi = (face_roi * 255).astype('uint8')
                
                threading.Thread(target=process_face_recognition, 
                               args=(face_roi, face_id)).start()
    
    # Draw faces with recognition results
    for face_data in faces_data:
        facial_area = face_data['facial_area']
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        
        with lock:
            result = get_closest_result(x, y, w, h, face_recognition_results)
        
        if result and result.get("matched"):
            name = result.get("name", "Unknown")
            distance = result.get("avg_distance", 0)
            draw_face_rectangle(frame, x, y, w, h, COLOR_GREEN, 
                              [f"{name} ({distance:.2f})"])
        else:
            draw_face_rectangle(frame, x, y, w, h, COLOR_RED, ["Unknown"])


# ============================================================================
# FEATURE FUNCTIONS - EMOTION RECOGNITION
# ============================================================================

def process_emotion_recognition(face_roi, face_id):
    """
    Process emotion recognition in a separate thread.
    
    Uses DeepFace to analyze emotions and age.
    Sets flag when starting and clears it when done to prevent concurrent threads.
    
    Args:
        face_roi: Cropped face image
        face_id: Unique identifier for this face position
    """
    global emotion_recognition_results, emotion_recognition_in_progress
    try:
        analysis = DeepFace.analyze(face_roi, actions=['emotion', 'age'], 
                                    enforce_detection=False)
        with lock:
            emotion_recognition_results[face_id] = {
                "emotion": analysis[0]['dominant_emotion'],
                "age": int(analysis[0]['age'])
            }
    except Exception as e:
        print(f"Emotion recognition error: {e}")
        with lock:
            emotion_recognition_results[face_id] = {"emotion": "Unknown", "age": 0}
    finally:
        # Always clear the in-progress flag when done
        with thread_lock:
            emotion_recognition_in_progress = False


def handle_emotion_recognition(frame, faces_data):
    """
    Handle emotion recognition mode - detect emotions on faces.
    
    Only launches a new emotion thread if no other is currently running.
    This prevents CPU/RAM overload from too many concurrent threads.
    
    Args:
        frame: Current video frame
        faces_data: List of detected faces from DeepFace
    """
    global frame_counter, emotion_recognition_in_progress
    
    # Process emotion recognition every N frames, but only if no thread is running
    if frame_counter % EMOTION_RECOGNITION_INTERVAL == 0 and faces_data:
        with thread_lock:
            if emotion_recognition_in_progress:
                # Skip this frame - previous recognition still in progress
                pass
            else:
                # Mark as in progress and launch thread for first face only
                emotion_recognition_in_progress = True
                face_data = faces_data[0]  # Process one face at a time
                face_roi = face_data['face']
                facial_area = face_data['facial_area']
                face_id = f"{facial_area['x']}_{facial_area['y']}_{facial_area['w']}_{facial_area['h']}"
                
                # Normalize face image if needed
                if face_roi.max() <= 1.0:
                    face_roi = (face_roi * 255).astype('uint8')
                
                threading.Thread(target=process_emotion_recognition, 
                               args=(face_roi, face_id)).start()
    
    # Draw faces with emotion results
    for face_data in faces_data:
        facial_area = face_data['facial_area']
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        
        with lock:
            result = get_closest_result(x, y, w, h, emotion_recognition_results)
        
        labels = []
        if result and result.get("emotion"):
            labels.append(f"{result['emotion']} (Age: {result['age']})")
        
        draw_face_rectangle(frame, x, y, w, h, COLOR_CYAN, labels if labels else None)


# ============================================================================
# FEATURE FUNCTIONS - ADD FACE
# ============================================================================

def handle_add_face_capture(frame, faces_data):
    """
    Handle face capture mode - capture and store face embeddings.
    
    Captures embeddings every N frames during the capture period.
    
    Args:
        frame: Current video frame
        faces_data: List of detected faces from DeepFace
    """
    global current_state, add_face_capture_count, frame_counter, add_face_start_frame
    
    elapsed_frames = frame_counter - add_face_start_frame
    
    # Check if capture period is complete
    if elapsed_frames >= ADD_FACE_DURATION_FRAMES:
        # Show completion message
        draw_text(frame, "=== FACE CAPTURE COMPLETE ===", 
                  (FRAME_WIDTH // 2 - 150, FRAME_HEIGHT // 2),
                  font_scale=0.8, text_color=COLOR_GREEN)
        draw_text(frame, f"Saved {add_face_capture_count} embeddings for '{add_face_name}'",
                  (FRAME_WIDTH // 2 - 180, FRAME_HEIGHT // 2 + 40))
        
        # Reset to idle state after showing completion
        if elapsed_frames >= ADD_FACE_DURATION_FRAMES + 60:  # Show for 1 second
            current_state = AppState.IDLE
            add_face_capture_count = 0
        return
    
    # Capture embedding every N frames
    if elapsed_frames % ADD_FACE_CAPTURE_INTERVAL == 0 and faces_data:
        face_data = faces_data[0]  # Use first detected face
        face_roi = face_data['face']
        facial_area = face_data['facial_area']
        
        # Normalize face image if needed
        if face_roi.max() <= 1.0:
            face_roi = (face_roi * 255).astype('uint8')
        
        # Insert embedding into database
        if pc.insert_single_embedding(cursor, conn, add_face_name, face_roi):
            add_face_capture_count += 1
            print(f"Captured embedding {add_face_capture_count} for {add_face_name}")
    
    # Draw faces with yellow rectangle during capture
    for face_data in faces_data:
        facial_area = face_data['facial_area']
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        draw_face_rectangle(frame, x, y, w, h, COLOR_YELLOW, [f"Capturing: {add_face_name}"])


def handle_add_face_name_input(key):
    """
    Handle keyboard input during name entry for adding a new face.
    
    Args:
        key: Key code from cv2.waitKey()
    
    Returns:
        bool: True if input was handled, False otherwise
    """
    global current_state, add_face_name, add_face_start_frame, frame_counter, add_face_capture_count
    
    if key == 27:  # ESC - cancel
        current_state = AppState.IDLE
        add_face_name = ""
        return True
    
    elif key == 13:  # ENTER - confirm name and start capture
        if add_face_name.strip():
            current_state = AppState.ADD_FACE_CAPTURE
            add_face_start_frame = frame_counter
            add_face_capture_count = 0
            return True
    
    elif key == 8:  # BACKSPACE - delete character
        add_face_name = add_face_name[:-1]
        return True
    
    elif 32 <= key <= 126:  # Printable ASCII characters
        add_face_name += chr(key)
        return True
    
    return False


# ============================================================================
# FEATURE FUNCTIONS - CLEAN DATABASE
# ============================================================================

def handle_clean_database(frame):
    """
    Clean the database by removing all stored face embeddings.
    
    Shows a confirmation message on the frame.
    
    Args:
        frame: Current video frame for displaying message
    """
    global current_state
    
    deleted_count = pc.clear_all_embeddings(cursor, conn)
    
    # Display confirmation
    draw_text(frame, "=== DATABASE CLEANED ===", 
              (FRAME_WIDTH // 2 - 120, FRAME_HEIGHT // 2),
              font_scale=0.8, text_color=COLOR_RED)
    draw_text(frame, f"Deleted {deleted_count} face embeddings",
              (FRAME_WIDTH // 2 - 130, FRAME_HEIGHT // 2 + 40))


# ============================================================================
# DEFAULT FACE DETECTION
# ============================================================================

def handle_idle_detection(frame, faces_data):
    """
    Handle default idle state - show blue rectangles around detected faces.
    
    Args:
        frame: Current video frame
        faces_data: List of detected faces from DeepFace
    """
    for face_data in faces_data:
        facial_area = face_data['facial_area']
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        draw_face_rectangle(frame, x, y, w, h, COLOR_BLUE)


# ============================================================================
# KEY HANDLING
# ============================================================================

def handle_key_input(key):
    """
    Handle keyboard input for state transitions.
    
    Args:
        key: Key code from cv2.waitKey()
    
    Returns:
        bool: True if application should quit, False otherwise
    """
    global current_state, face_recognition_results, emotion_recognition_results, add_face_name
    
    if key == ord('q') or key == ord('Q'):
        return True  # Quit application
    
    # Handle name input separately when in ADD_FACE_NAME state
    if current_state == AppState.ADD_FACE_NAME:
        handle_add_face_name_input(key)
        return False
    
    # Handle state transitions from IDLE
    if current_state == AppState.IDLE:
        if key == ord('1'):
            current_state = AppState.FACE_RECOGNITION
            face_recognition_results.clear()
            print("Face Recognition activated")
        elif key == ord('2'):
            current_state = AppState.EMOTION_RECOGNITION
            emotion_recognition_results.clear()
            print("Emotion Recognition activated")
        elif key == ord('3'):
            current_state = AppState.ADD_FACE_NAME
            add_face_name = ""
            print("Add Face - Enter name")
        elif key == ord('4'):
            # Clean database is instant, show message for a moment
            return False  # Will be handled in main loop
    
    # Handle state transitions back to IDLE
    elif current_state == AppState.FACE_RECOGNITION:
        if key == ord('1'):
            current_state = AppState.IDLE
            print("Face Recognition deactivated")
    
    elif current_state == AppState.EMOTION_RECOGNITION:
        if key == ord('2'):
            current_state = AppState.IDLE
            print("Emotion Recognition deactivated")
    
    return False


# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    """
    Main application loop.
    
    Handles video capture, face detection, and state-based processing.
    """
    global frame_counter, current_state
    
    # Initialize camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Variables for clean database confirmation display
    show_clean_message = False
    clean_message_frame = 0
    clean_deleted_count = 0
    
    print("=== Face Recognition System Started ===")
    print("Press Q to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Detect faces in current frame
        try:
            faces_data = DeepFace.extract_faces(frame, enforce_detection=False)
        except Exception:
            faces_data = []
        
        # Handle clean database message display
        if show_clean_message:
            if frame_counter - clean_message_frame < 90:  # Show for 1.5 seconds
                draw_text(frame, "=== DATABASE CLEANED ===", 
                          (FRAME_WIDTH // 2 - 120, FRAME_HEIGHT // 2),
                          font_scale=0.8, text_color=COLOR_RED)
                draw_text(frame, f"Deleted {clean_deleted_count} face embeddings",
                          (FRAME_WIDTH // 2 - 130, FRAME_HEIGHT // 2 + 40))
            else:
                show_clean_message = False
        
        # Process based on current state
        if current_state == AppState.IDLE:
            handle_idle_detection(frame, faces_data)
        
        elif current_state == AppState.FACE_RECOGNITION:
            handle_face_recognition(frame, faces_data)
        
        elif current_state == AppState.EMOTION_RECOGNITION:
            handle_emotion_recognition(frame, faces_data)
        
        elif current_state == AppState.ADD_FACE_NAME:
            handle_idle_detection(frame, faces_data)  # Still show blue rectangles
        
        elif current_state == AppState.ADD_FACE_CAPTURE:
            handle_add_face_capture(frame, faces_data)
        
        # Draw instruction overlay
        draw_instructions(frame, current_state)
        
        # Display frame
        cv2.imshow("Face Recognition System", frame)
        
        # Handle keyboard input
        frame_counter += 1
        key = cv2.waitKey(1) & 0xFF
        
        if key != 255:  # A key was pressed
            # Special handling for clean database
            if current_state == AppState.IDLE and key == ord('4'):
                clean_deleted_count = pc.clear_all_embeddings(cursor, conn)
                show_clean_message = True
                clean_message_frame = frame_counter
            elif handle_key_input(key):
                break  # Quit
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("=== Face Recognition System Stopped ===")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()