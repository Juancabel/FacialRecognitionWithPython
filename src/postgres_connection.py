import psycopg2
import os
from deepface import DeepFace
import cv2
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================

# Available models:
# "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",
# "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet", "Buffalo_L"
MODEL_NAME = "Facenet512"  # Change this to use a different model
DISTANCE_THRESHOLD = 15  # Adjust based on model (Facenet512: 20, Facenet: 10, SFace: 1.0)


def connect_db():
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="deepface",
        user="postgres",
        password="password"
    )
    
    cursor = conn.cursor()
    return cursor , conn

def reset_table(cursor,conn):
    cursor.execute("drop table if exists embeddings")
    cursor.execute("create table embeddings (name varchar, embedding numeric[])")

def insert_images_from_file(cursor,conn,path):
    representations = []
    print(f"Scanning directory: {path}")
    for dirpath, dirnames, filenames in os.walk(path):
        print(f"Found {len(filenames)} files in {dirpath}: {filenames}")
        for filename in filenames:
            img_path = os.path.join(dirpath, filename)
            print(f"Checking: {img_path}")
            if ".jpg" in img_path.lower() or ".jpeg" in img_path.lower() or ".png" in img_path.lower():
                try:
                    print(f"Processing: {img_path}")
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Failed to read image: {img_path}")
                        continue
                    
                    faces_data = DeepFace.extract_faces(img, detector_backend="mtcnn")
                    
                    if not faces_data:
                        print(f"No faces found in {img_path}")
                        continue
                    
                    face_img = faces_data[0]["face"]
                    obj = DeepFace.represent(
                        img_path=face_img,
                        model_name=MODEL_NAME,
                        detector_backend="skip"  
                    )
                    embedding = obj[0]["embedding"]
                    representations.append((img_path, embedding))
                    print(f"Processed: {img_path}")
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    print(f"Total images to insert: {len(representations)}")
    for img_path, embedding in representations:
        cursor.execute(
            "INSERT INTO embeddings (name, embedding) VALUES (%s, %s)",
            (img_path.split("\\")[-1].split("/")[-1], embedding)
        )
    conn.commit()


def get_representation(img):
    """
    Get the face embedding/representation for an image.
    
    Uses the configured model (Facenet by default for speed).
    
    Args:
        img: Face image (numpy array)
    
    Returns:
        list: Face embedding vector
    """
    target = DeepFace.represent(
        img_path=img,
        model_name=MODEL_NAME,
        detector_backend="skip"
    )[0]["embedding"]
    return target


def search_similar_faces(cursor, conn, target):
    """
    Search for similar faces in the database using Euclidean distance.
    
    Returns the most common name among all embeddings below the threshold,
    which improves accuracy when multiple embeddings exist per person.
    
    Args:
        cursor: Database cursor
        conn: Database connection
        target: Target embedding vector to search for
    
    Returns:
        dict: Contains name, avg_distance, and matched status
    """
    # First, calculate distance for each individual embedding row
    # Then filter by threshold
    query = f"""
        SELECT name, distance
        FROM (
            SELECT name, sqrt(sum(sq_diff)) as distance
            FROM (
                SELECT name, ctid, pow(unnest(embedding) - unnest(ARRAY{target}), 2) as sq_diff
                FROM embeddings
            ) element_diffs
            GROUP BY name, ctid
        ) row_distances
        WHERE distance < {DISTANCE_THRESHOLD}
        ORDER BY distance
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    
    if len(rows) == 0:
        return {
            "name": None,
            "avg_distance": None,
            "matched": False
        }
    
    # Extract base names (remove suffix like "_1", "_2" for multiple captures)
    names = []
    distances = []
    for row_name, dist in rows:
        name_parts = row_name.split("_")
        if len(name_parts) > 1 and name_parts[-1].isdigit():
            base_name = "_".join(name_parts[:-1])
        else:
            base_name = row_name
        names.append(base_name)
        distances.append(float(dist))
    
    # Find the most common name among all matches
    name_counts = Counter(names)
    most_common_name = name_counts.most_common(1)[0][0]
    
    # Get the average distance for the most common name
    distances_for_name = [dist for name, dist in zip(names, distances) 
                          if name == most_common_name]
    avg_distance = sum(distances_for_name) / len(distances_for_name)
    
    return {
        "name": most_common_name,
        "avg_distance": float(avg_distance),
        "matched": True
    }


def insert_single_embedding(cursor, conn, name, face_img):
    """
    Insert a single face embedding into the database.
    
    Args:
        cursor: Database cursor
        conn: Database connection
        name: Name to associate with this face
        face_img: Face image (already extracted/cropped)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        obj = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            detector_backend="skip"
        )
        embedding = obj[0]["embedding"]
        cursor.execute(
            "INSERT INTO embeddings (name, embedding) VALUES (%s, %s)",
            (name, embedding)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error inserting embedding: {e}")
        return False


def clear_all_embeddings(cursor, conn):
    """
    Delete all embeddings from the database.
    
    Args:
        cursor: Database cursor
        conn: Database connection
    
    Returns:
        int: Number of rows deleted
    """
    try:
        cursor.execute("DELETE FROM embeddings")
        deleted_count = cursor.rowcount
        conn.commit()
        print(f"Deleted {deleted_count} embeddings from database")
        return deleted_count
    except Exception as e:
        print(f"Error clearing embeddings: {e}")
        conn.rollback()
        return 0


def get_embedding_count(cursor):
    """
    Get the total number of embeddings in the database.
    
    Args:
        cursor: Database cursor
    
    Returns:
        int: Number of embeddings
    """
    cursor.execute("SELECT COUNT(*) FROM embeddings")
    return cursor.fetchone()[0]