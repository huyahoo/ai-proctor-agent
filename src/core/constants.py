# Constants for keypoint indexing (MediaPipe BlazePose)
POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), # Arms
                    (9, 10), (11, 12), (12, 13), (13, 14), (15, 16), (17, 18), # Legs (simplified)
                    (11, 23), (12, 24), (23, 24)] # Torso (simplified)

# Object classes for YOLO (expand if needed)
# Assuming YOLOv8n, common objects: person, cell phone, book, laptop, bottle, etc.
# You might need to train a custom YOLO model for specific "illegal materials" like specific notes.
YOLO_CLASSES_OF_INTEREST = ['cell phone', 'book', 'laptop', 'person'] # Add more specific ones if you fine-tune YOLO

