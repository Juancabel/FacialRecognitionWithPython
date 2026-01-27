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
        user="postgres2",
        password="password"
    )
    
    cursor = conn.cursor()
    return cursor , conn

def reset_table(cursor):
    cursor.execute("drop table if exists embeddings")
    cursor.execute("create table embeddings (name varchar, embedding decimal[])")

def insert_images_from_file(cursor,conn,path):
    representations = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            img_path = f"{dirpath}{filename}"
            if ".jpg" in img_path:
                obj = DeepFace.represent(
                    img_path=img_path, 
                    model_name="Facenet", 
                    detector_backend="mtcnn",
                )
                embedding = obj[0]["embedding"]
                representations.append((img_path, embedding))

    for img_path, embedding in representations:
        statement = f"""
        insert into 
        embeddings 
        (name, embedding) 
        values 
        ("{img_path}", ARRAY{embedding});
        """
        cursor.execute(statement)
    conn.commit()


def get_representation(img):
#target_path = "target.jpg"
#target_img = cv2.imread(target_path)
 
    target = DeepFace.represent(
        img_path=img,
        #model_name="Facenet",
        detector_backend="mtcnn"
    )[0]["embedding"]
    return target


def search_similar_faces(cursor,conn,target):
    
    threshold = 10
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
    
    for img_path, distance in rows:
        print(img_path, distance)
        img = cv2.imread(img_path)
        
        fig = plt.figure(figsize = (7, 7))
        
        fig.add_subplot(1, 2, 1)
        plt.imshow(img[:,:,::-1])
        
        fig.add_subplot(1, 2, 2)
        plt.imshow(img[:,:,::-1])
        
        plt.show()