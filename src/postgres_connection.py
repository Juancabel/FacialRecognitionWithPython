import psycopg2
import os
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt


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
                        model_name="Facenet512",
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
#target_path = "target.jpg"
#target_img = cv2.imread(target_path)
 
    target = DeepFace.represent(
        img_path=img,
        model_name="Facenet512",
        detector_backend="skip"
    )[0]["embedding"]
    return target


def search_similar_faces(cursor,conn,target):
    
    threshold = 20
    query = f"""
        select name, distance
        from (
            select name, sqrt(sum(distance)) as distance
            from (
                select name, pow(unnest(embedding) - unnest(ARRAY{target}), 2) as distance
                from embeddings
            ) sq
            group by name
        ) sq2
        where distance < {threshold}
        order by distance
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    print(len(rows))
    
    if len(rows)==0:
        return {
            "name": None,
            "avg_distance": None,
            "matched": False
        }
    
    img_path, distance = rows[0]
    name = img_path.split("_")
    if len(name) > 1:
        name = "_".join(name[:-1])
    else:
        name = img_path
    
    return {
        "name": name,
        "avg_distance": float(distance),
        "matched": True
    }