import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np


model = YOLO('Modules/Object Detection/best.pt').to('mps')
cap = cv2.VideoCapture(0)
track_history = defaultdict(lambda: [])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.track(frame, persist=True, device='mps', verbose=False)
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)
            
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, 
                         color=(230, 0, 0), thickness=2)
    
    annotated_frame = results[0].plot()
    cv2.imshow('Tracking', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()