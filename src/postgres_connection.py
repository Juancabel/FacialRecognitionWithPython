import psycopg2
import os
from deepface import DeepFace
import cv2
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# Available models:
# "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",
# "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet", "Buffalo_L"
MODEL_NAME = "Facenet512"  # Change this to use a different model
DISTANCE_THRESHOLD = 15  # Adjust based on model (Facenet512: 20, Facenet: 10, SFace: 1.0)

# In-memory cache for fast searching
_embeddings_cache = None  # Will store {"names": [...], "embeddings": np.array}
_cache_dirty = True  # Flag to indicate cache needs refresh


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
    invalidate_cache()  # Mark cache as dirty after bulk insert


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


def load_embeddings_cache(cursor):
    """
    Load all embeddings from database into memory for fast searching.
    
    This is much faster than querying the database every time since
    NumPy operations are highly optimized.
    
    Args:
        cursor: Database cursor
    
    Returns:
        dict: Contains 'names' list and 'embeddings' numpy array
    """
    global _embeddings_cache, _cache_dirty
    
    if _embeddings_cache is not None and not _cache_dirty:
        return _embeddings_cache
    
    cursor.execute("SELECT name, embedding FROM embeddings")
    rows = cursor.fetchall()
    
    if not rows:
        _embeddings_cache = {"names": [], "embeddings": np.array([])}
        _cache_dirty = False
        return _embeddings_cache
    
    names = []
    embeddings = []
    
    for name, embedding in rows:
        names.append(name)
        # Convert from Decimal array to float array
        embeddings.append([float(x) for x in embedding])
    
    _embeddings_cache = {
        "names": names,
        "embeddings": np.array(embeddings, dtype=np.float32)
    }
    _cache_dirty = False
    print(f"Loaded {len(names)} embeddings into cache")
    return _embeddings_cache


def invalidate_cache():
    """
    Mark the cache as dirty so it will be reloaded on next search.
    Call this after inserting or deleting embeddings.
    """
    global _cache_dirty
    _cache_dirty = True


def search_similar_faces_fast(cursor, target):
    """
    Fast in-memory search for similar faces using NumPy.
    
    This is ~100x faster than the SQL-based approach because:
    1. Embeddings are cached in memory (no DB round-trip)
    2. NumPy uses vectorized operations (SIMD optimized)
    
    Args:
        cursor: Database cursor (for loading cache if needed)
        target: Target embedding vector to search for
    
    Returns:
        dict: Contains name, avg_distance, and matched status
    """
    cache = load_embeddings_cache(cursor)
    
    if len(cache["names"]) == 0:
        return {
            "name": None,
            "avg_distance": None,
            "matched": False
        }
    
    # Convert target to numpy array
    target_np = np.array(target, dtype=np.float32)
    
    # Compute Euclidean distances using vectorized operations
    # This is extremely fast compared to SQL UNNEST approach
    distances = np.linalg.norm(cache["embeddings"] - target_np, axis=1)
    
    # Find minimum distance
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    
    if min_distance >= DISTANCE_THRESHOLD:
        return {
            "name": None,
            "avg_distance": float(min_distance),
            "matched": False
        }
    
    # Extract name (remove suffix like "_1", "_2" for multiple captures)
    name = cache["names"][min_idx]
    name_parts = name.split("_")
    if len(name_parts) > 1 and name_parts[-1].isdigit():
        name = "_".join(name_parts[:-1])
    
    return {
        "name": name,
        "avg_distance": float(min_distance),
        "matched": True
    }


def search_similar_faces(cursor, conn, target):
    """
    Search for similar faces - wrapper that uses fast in-memory search.
    
    Kept for backward compatibility. Now delegates to search_similar_faces_fast.
    
    Args:
        cursor: Database cursor
        conn: Database connection (not used, kept for compatibility)
        target: Target embedding vector to search for
    
    Returns:
        dict: Contains name, avg_distance, and matched status
    """
    return search_similar_faces_fast(cursor, target)


def insert_single_embedding(cursor, conn, name, face_img):
    """
    Insert a single face embedding into the database.
    
    Also invalidates the cache so the new embedding will be searchable.
    
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
        invalidate_cache()  # Mark cache as dirty
        return True
    except Exception as e:
        print(f"Error inserting embedding: {e}")
        return False


def clear_all_embeddings(cursor, conn):
    """
    Delete all embeddings from the database.
    
    Also invalidates the cache.
    
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
        invalidate_cache()  # Mark cache as dirty
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