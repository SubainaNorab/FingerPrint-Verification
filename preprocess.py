
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model (tiny, fast)
model = YOLO('yolov8n.pt')  

def preprocess(img, show_crop=False, show_final=False):
   
    results = model(img)[0]  # detecting finger

    
    if len(results.boxes) == 0:
        print("No finger detected, using full image")
        cropped = img
    else:
        
        box = results.boxes.xyxy[0].cpu().numpy()  
        x1, y1, x2, y2 = box.astype(int)
        cropped = img[y1:y2, x1:x2]

    if show_crop:
        cv2.imshow("Finger Crop", cropped)
        cv2.waitKey(0)

    # Grayscale-->remove noise 
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # enhance ridges
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    
    final = cv2.resize(enhanced, (224,224))
    final_rgb = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB) / 255.0

    if show_final:
        cv2.imshow("Final Processed 224x224", (final_rgb*255).astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_rgb


