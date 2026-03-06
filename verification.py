import cv2
import numpy as np
import pickle
# from models import cnn_model

from models import get_embedding
# Load database
with open("fingerData.pkl", "rb") as f:
    DB = pickle.load(f)



def verify(frame, threshold=0.85):  
    emb_test = get_embedding(frame)
    best_score = -1  
    match_user = None

    for user, emb_list in DB.items():
        for emb in emb_list:
            dist = np.dot(emb_test, emb) / (np.linalg.norm(emb_test) * np.linalg.norm(emb))
            if dist > best_score:
                best_score = dist
                match_user = user

    if best_score > threshold:
        return f"User: {match_user} (score: {best_score:.3f})"
    else:
        return f"Not Found (best score: {best_score:.3f})"


image_path = 'shaT.jpg'
frame = cv2.imread(image_path)
if frame is None:
    print("Image not found!")
    exit()

result = verify(frame)
print(result)

# cv2.putText(frame, result, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if "User" in result else (0,0,255), 2)
# cv2.imshow("Verification", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
