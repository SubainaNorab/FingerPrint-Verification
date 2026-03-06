import os
import cv2
from models import get_embedding
import pickle
import numpy as np

with open("fingerData.pkl", "rb") as f:
    DB = pickle.load(f)

THRESHOLD = 0.8

test_folder = "test_images"

pairs = []
labels = []

users = os.listdir(test_folder)
for user in users:
    user_folder = os.path.join(test_folder, user)
    if not os.path.isdir(user_folder): continue
    user_images = [os.path.join(user_folder, f) for f in os.listdir(user_folder)]
    
    # same user
    for i in range(len(user_images)):
        for j in range(i+1, len(user_images)):
            pairs.append((user_images[i], user_images[j]))
            labels.append(1)
    
    # different users
    for other_user in users:
        if other_user == user: continue
        other_folder = os.path.join(test_folder, other_user)
        other_images = [os.path.join(other_folder, f) for f in os.listdir(other_folder)]
        for img1 in user_images:
            for img2 in other_images:
                pairs.append((img1, img2))
                labels.append(0)


def verify_pair(img_path1, img_path2, threshold=THRESHOLD):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    if img1 is None or img2 is None:
        return False
    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return 1 if sim >= threshold else 0


thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

results = []

for THRESHOLD in thresholds:
    TP = TN = FP = FN = 0
    for (img1, img2), label in zip(pairs, labels):
        pred = verify_pair(img1, img2, threshold=THRESHOLD)
        if label == 1 and pred == 1: TP += 1
        if label == 1 and pred == 0: FN += 1
        if label == 0 and pred == 0: TN += 1
        if label == 0 and pred == 1: FP += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    results.append((THRESHOLD, accuracy, tpr, fpr))

best = max(results, key=lambda x: x[1])
best_threshold, best_acc, best_tpr, best_fpr = best

print("Threshold results summary:")
for r in results:
    print(f"Threshold: {r[0]:.2f} → Accuracy: {r[1]:.3f}, TPR: {r[2]:.3f}, FPR: {r[3]:.3f}")

print("\nBest Threshold:")
print(f"Threshold: {best_threshold:.2f}, Accuracy: {best_acc:.3f}, TPR: {best_tpr:.3f}, FPR: {best_fpr:.3f}")
