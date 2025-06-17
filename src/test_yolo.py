from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("./models/yolov8n.pt")  # Replace with your model path
YOLO_CLASSES_OF_INTEREST = ['earphone', 'person', 'cell phone', 'book', 'laptop', 'bottle', 'backpack', 'keyboard', 'mouse', 'tv', 'remote', 'clock', 'scissors', 'pen', 'pencil', 'tablet', 'calculator', 'cheatsheet', 'Smart Watch', 'mobile-phone', "paper"]


def detect(model, frame: np.ndarray) -> list:
    """
    Detects objects in a frame using YOLOv8.
    Returns a list of dictionaries:
    [{'bbox': [x1, y1, x2, y2], 'label': 'person', 'confidence': 0.95}, ...]
    """
    if model is None:
        return []

    results = model(frame, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            confidence = box.conf.item()
            class_id = box.cls.item()
            label = model.names[int(class_id)]
            print(f"Detected {label} with confidence {confidence}")

            if confidence > 0.5 and label in YOLO_CLASSES_OF_INTEREST:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'label': label,
                    'confidence': confidence
                })

    return detections

image = cv2.imread("images/IMG_4737.jpg")  # Replace with your image path
detections = detect(model, image)
print(f"Detected {len(detections)} objects: ", detections)
for detection in detections:
    bbox = detection['bbox']
    label = detection['label']
    confidence = detection['confidence']
    print(f"Detected {label} with confidence {confidence} at {bbox}")
    
    # Convert bbox coordinates to integers
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw bounding box and label
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
cv2.imwrite("results/output.jpg", image) # Save the output image with detections