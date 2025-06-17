# OpenPifPaf COCO skeleton connections (1-based indices)
COCO_PERSON_SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13),  # Legs
    (12, 13), (6, 12), (7, 13),              # Hips and torso
    (6, 7), (6, 8), (7, 9),                  # Shoulders and arms
    (8, 10), (9, 11),                        # Forearms
    (2, 3), (1, 2), (1, 3),                  # Face
    (2, 4), (3, 5), (4, 6), (5, 7)           # Ears to shoulders
]

# Convert to 0-based indices for easier use with list of keypoints
COCO_PERSON_SKELETON_INDICES = [(j1-1, j2-1) for j1, j2 in COCO_PERSON_SKELETON]

# Object classes for YOLO (expand if needed for specific items)
# These are common COCO dataset classes. You might need to fine-tune YOLO for specific "cheating notes" etc.
YOLO_CLASSES_OF_INTEREST = ['person', 'cell phone', 'book', 'laptop', 'bottle', 'backpack', 'keyboard', 'mouse', 'tv', 'remote', 'clock', 'scissors', 'pen', 'pencil', 'tablet', 'calculator']
UNAUTHORIZED_ITEMS = ['cheatsheet', 'Smartwatch', 'Phone', 'earphone']