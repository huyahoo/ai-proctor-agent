import json
import os
import cv2
from core.config import Config
from PIL import Image
import numpy as np
import time # For unique ID

class FeedbackLearner:
    def __init__(self, config: Config):
        self.config = config
        os.makedirs(self.config.FEEDBACK_DATA_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.config.FEEDBACK_DATA_DIR, "images"), exist_ok=True)
        self.annotation_file_path = self.config.FEEDBACK_ANNOTATIONS_FILE
        print(f"Feedback data will be stored in: {self.config.FEEDBACK_DATA_DIR}")

    def save_feedback(self, event_data: dict, frame_sequence: list, vlm_decision: str, human_feedback: str, vlm_explanation: str):
        """
        Saves human feedback and associated data to a JSONL file.
        Also saves a representative frame for potential future VLM fine-tuning.
        Args:
            event_data (dict): The original anomaly event data.
            frame_sequence (list): List of PIL Image objects of the frames used for VLM analysis.
            vlm_decision (str): The VLM's decision (e.g., "Cheating Confirmed").
            human_feedback (str): The human's feedback ("confirmed_cheating" or "false_positive").
            vlm_explanation (str): The VLM's explanation text.
        """

        image_save_path = None
        if frame_sequence:
            mid_frame_idx = len(frame_sequence) // 2
            representative_frame = frame_sequence[mid_frame_idx]

            # Generate a unique filename for the image, robust to special chars
            event_id_clean = str(event_data.get('event_id', f"event_{int(time.time()*1000)}")).replace('.', '_').replace(':', '_').replace('/', '_').replace('\\', '_').replace(' ', '_')
            frame_filename = f"{event_id_clean}_{human_feedback}.jpg"
            image_save_path_full = os.path.join(self.config.FEEDBACK_DATA_DIR, "images", frame_filename)

            try:
                representative_frame.save(image_save_path_full)
                # Store relative path from FEEDBACK_DATA_DIR for dataset portability
                image_save_path = os.path.relpath(image_save_path_full, self.config.FEEDBACK_DATA_DIR)
            except Exception as e:
                print(f"Warning: Could not save representative frame {image_save_path_full}: {e}")

        annotation = {
            "event_id": event_data.get('event_id', f"event_{int(time.time()*1000)}"),
            "video_path": event_data.get('video_path'),
            "timestamp_sec": event_data.get('timestamp'),
            "event_type": event_data.get('type'),
            "person_ids": event_data.get('person_ids', []),
            "object_label": event_data.get('object_label', ''),
            "vlm_initial_decision": vlm_decision,
            "vlm_explanation": vlm_explanation,
            "human_feedback": human_feedback,
            "image_path_in_dataset": image_save_path
        }

        with open(self.annotation_file_path, 'a') as f:
            f.write(json.dumps(annotation) + '\n')
        print(f"Feedback saved for event '{annotation['event_id']}': {human_feedback}")

    def prepare_for_finetuning(self):
        """
        Reads the collected feedback data and prepares it into a VLM-friendly
        format (e.g., LLaVA's JSONL conversation format).
        This method would be called periodically to generate a new fine-tuning dataset.
        For a hackathon, this is more conceptual and shows the data format.
        """
        print(f"Preparing data from {self.annotation_file_path} for VLM fine-tuning...")
        finetuning_data = []
        if not os.path.exists(self.annotation_file_path):
            print("No feedback annotations found yet.")
            return []

        with open(self.annotation_file_path, 'r') as f:
            for line in f:
                annotation = json.loads(line)

                # Only include entries that have a saved image and human feedback (confirmed or false positive)
                abs_image_path = None
                if annotation.get('image_path_in_dataset'):
                    abs_image_path = os.path.join(self.config.FEEDBACK_DATA_DIR, annotation['image_path_in_dataset'])

                if abs_image_path and os.path.exists(abs_image_path) and annotation['human_feedback'] in ["confirmed_cheating", "false_positive"]:

                    user_prompt = f"Analyze this exam scene for cheating. The initial alert was for {annotation['event_type'].replace('_', ' ')}."
                    if annotation['person_ids']:
                        user_prompt += f" Involving persons: {', '.join(annotation['person_ids'])}."
                    if annotation['object_label']:
                        user_prompt += f" An object '{annotation['object_label']}' was also detected."
                    user_prompt += " Is this cheating? Provide a detailed explanation."

                    assistant_response = ""
                    if annotation['human_feedback'] == "confirmed_cheating":
                        assistant_response = f"Decision: Cheating Confirmed. Explanation: The students exhibited {annotation['event_type'].replace('_', ' ')}. Visual evidence from the frames confirms the suspicious activity based on observed actions matching the proctoring rules. Specifically, {annotation['vlm_explanation']}"
                    elif annotation['human_feedback'] == "false_positive":
                        assistant_response = f"Decision: Not Cheating. Explanation: Although an alert was raised for {annotation['event_type'].replace('_', ' ')}, careful visual review of the frames shows no cheating. The actions observed were legitimate, for example: {annotation['vlm_explanation']}"

                    finetuning_data.append({
                        "image": abs_image_path, # Absolute path required for LLaVA training scripts
                        "conversations": [
                            {"role": "user", "content": user_prompt + " <image>"},
                            {"role": "assistant", "content": assistant_response}
                        ]
                    })

        output_finetune_path = os.path.join(self.config.FEEDBACK_DATA_DIR, "vlm_finetune_dataset.jsonl")
        with open(output_finetune_path, 'w') as f:
            for entry in finetuning_data:
                f.write(json.dumps(entry) + '\n')
        print(f"Prepared {len(finetuning_data)} samples for VLM fine-tuning in {output_finetune_path}")
        return finetuning_data

