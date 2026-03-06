import cv2
import numpy as np
import os
import pickle
from preprocess import preprocess
from models import get_embedding

# cnn = cnn_model()
DB = {}
dataset_folder = 'images'



for user in os.listdir(dataset_folder):
    user_folder = os.path.join(dataset_folder, user)
    if not os.path.isdir(user_folder): continue

    DB[user] = []

    for img_name in os.listdir(user_folder):
        img_path = os.path.join(user_folder, img_name)
        img = cv2.imread(img_path)
        if img is None: continue

        # Preprocess
        # preprocessed_img = preprocess(img)

        
        emb = get_embedding(img)
        DB[user].append(emb.tolist())

        



# Save
with open('fingerData.pkl', 'wb') as f:
    pickle.dump(DB, f)
print("Database created")







