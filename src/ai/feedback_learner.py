import json
import os
import cv2
from core.config import Config
from PIL import Image
import numpy as np

class FeedbackLearner:
    def __init__(self, config: Config):
        self.config = config
        os.makedirs(self.config.FEEDBACK_DATA_DIR, exist_ok=True)
        self.annotation_file_path = self.config.FEEDBACK_ANNOTATIONS_FILE
        print(f"Feedback data will be stored in: {self.config.FEEDBACK_DATA_DIR}")

    def save_feedback(self, event_data: dict, frame_sequence: list, vlm_decision: str, human_feedback: str, vlm_explanation: str):
        """
        Saves human feedback and associated data to a JSONL file.
        Also saves relevant frames for potential future VLM fine-tuning.
        """

        # Save a representative frame (e.g., middle frame)
        if frame_sequence:
            # Ensure frame_sequence contains PIL Images or convertible objects
            mid_frame_idx = len(frame_sequence) // 2
            representative_frame = frame_sequence[mid_frame_idx]

            # Generate a unique filename for the image
            frame_filename = f"event_{event_data.get('timestamp', 'unknown').replace('.', '_')}_{human_feedback}.jpg"
            image_save_path = os.path.join(self.config.FEEDBACK_DATA_DIR, "images", frame_filename)
            os.makedirs(os.path.join(self.config.FEEDBACK_DATA_DIR, "images"), exist_ok=True)

            if isinstance(representative_frame, Image.Image):
                representative_frame.save(image_save_path)
            elif isinstance(representative_frame, np.ndarray):
                cv2.imwrite(image_save_path, representative_frame)
            else:
                print("Warning: Representative frame not a PIL Image or numpy array, cannot save.")
                image_save_path = None # Don't record a path if saving failed
        else:
            image_save_path = None

        annotation = {
            "event_id": f"event_{cv2.getTickCount()}", # Unique ID for this event
            "video_path": event_data.get('video_path'),
            "timestamp_sec": event_data.get('timestamp'),
            "event_type": event_data.get('type'),
            "person_ids": event_data.get('person_ids'),
            "object_label": event_data.get('object_label'),
            "vlm_initial_decision": vlm_decision,
            "vlm_explanation": vlm_explanation,
            "human_feedback": human_feedback, # "confirmed_cheating", "false_positive", "false_negative"
            "image_path_in_dataset": image_save_path # Path relative to FEEDBACK_DATA_DIR/images
        }

        with open(self.annotation_file_path, 'a') as f:
            f.write(json.dumps(annotation) + '\n')
        print(f"Feedback saved for event at {event_data.get('timestamp')}s: {human_feedback}")

    def prepare_for_finetuning(self):
        """
        Reads the collected feedback data and prepares it into a VLM-friendly
        format (e.g., LLaVA's JSONL conversation format).
        This method would be called periodically to generate a new fine-tuning dataset.
        For a hackathon, this is more conceptual.
        """
        print(f"Preparing data from {self.annotation_file_path} for VLM fine-tuning...")
        finetuning_data = []
        if not os.path.exists(self.annotation_file_path):
            print("No feedback annotations found yet.")
            return []

        with open(self.annotation_file_path, 'r') as f:
            for line in f:
                annotation = json.loads(line)

                # Construct a conversation entry suitable for LLaVA-like models
                # This is a simplified example; real VLM training data is more complex.
                if annotation['image_path_in_dataset'] and annotation['human_feedback'] in ["confirmed_cheating", "false_positive"]:
                    user_prompt = f"Analyze this exam scene for cheating. The initial alert was for {annotation['event_type']} involving {', '.join(annotation['person_ids'])}."

                    assistant_response = ""
                    if annotation['human_feedback'] == "confirmed_cheating":
                        assistant_response = f"Cheating confirmed. The students exhibited {annotation['event_type'].replace('_', ' ')}. Visual evidence: {annotation['vlm_explanation']}"
                    elif annotation['human_feedback'] == "false_positive":
                        assistant_response = f"Not cheating. Although an alert was raised for {annotation['event_type'].replace('_', ' ')}, careful review shows no cheating. Visual evidence indicates: {annotation['vlm_explanation']}"

                    finetuning_data.append({
                        "image": os.path.abspath(annotation['image_path_in_dataset']), # Absolute path for fine-tuning script
                        "conversations": [
                            {"role": "user", "content": user_prompt + " <image>"},
                            {"role": "assistant", "content": assistant_response}
                        ]
                    })

        # Save this generated data to a new JSONL file for fine-tuning
        output_finetune_path = os.path.join(self.config.FEEDBACK_DATA_DIR, "vlm_finetune_dataset.jsonl")
        with open(output_finetune_path, 'w') as f:
            for entry in finetuning_data:
                f.write(json.dumps(entry) + '\n')
        print(f"Prepared {len(finetuning_data)} samples for VLM fine-tuning in {output_finetune_path}")
        return finetuning_data

