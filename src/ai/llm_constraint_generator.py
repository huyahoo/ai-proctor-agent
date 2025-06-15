import google.generativeai as genai
import re
from core.config import Config
from core.logger import logger

class LLMConstraintGenerator:
    def __init__(self, config: Config):
        self.config = config
        genai.configure(api_key=self.config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(self.config.LLM_MODEL_NAME)
        logger.success(f"LLM model '{self.config.LLM_MODEL_NAME}' initialized for constraint generation.")

    def generate_constraints(self, event_data: dict) -> str:
        """
        Generates dynamic constraints for the VLM based on a detected anomaly event.
        Args:
            event_data (dict): A dictionary describing the initial anomaly event.
        Returns:
            str: A natural language string of constraints for the VLM.
        """
        prompt = self._build_prompt(event_data)
        try:
            response = self.model.generate_content(prompt)
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                logger.warning("LLM response did not contain expected content or was blocked.")
                return "Analyze for any further suspicious activity in the video segment."
        except Exception as e:
            logger.error(f"Error calling LLM for constraint generation: {e}")
            return f"Analyze for any further suspicious activity. LLM error: {e}"

    def _build_prompt(self, event_data: dict) -> str:
        """Constructs the prompt for the LLM based on event data."""
        event_type = event_data.get('type', 'unknown_event')
        person_ids = event_data.get('person_ids', [])
        object_label = event_data.get('object_label', '')

        base_prompt = "An initial suspicious event has been detected during an in-person exam. Based on the event type and details, generate a concise, specific instruction or a set of 'if-then' verification constraints for a Vision-Language Model (VLM) to confirm if actual cheating is occurring in the provided video frames. Focus on observable actions, interactions, and temporal patterns within the brief segment.\n\n"

        if event_type == 'collaborative_gaze_correlation':
            p1, p2 = person_ids[0], person_ids[1] if len(person_ids) > 1 else 'another student'
            return (f"{base_prompt}Initial event: Sustained, correlated side-gaze detected between {p1} and {p2}.\n"
                    "Constraint: If the video shows either student repeatedly glancing at the other's paper for more than 2 seconds, OR if either student makes a distinct non-verbal signal (e.g., pointing, specific hand gestures) towards the other's paper or screen, OR if any material (e.g., a note, small object) is exchanged between them, then confirm 'Collaborative Cheating: Peeking/Signaling'. Otherwise, consider it 'Not Cheating: Accidental Glance'.")

        elif event_type == 'individual_unauthorized_material':
            p_id = person_ids[0] if person_ids else 'a student'
            return (f"{base_prompt}Initial event: {p_id} detected with a potential unauthorized item ('{object_label}') in their immediate vicinity (lap, desk, or hand).\n"
                    "Constraint: If the video shows the student actively interacting with the detected item (e.g., reading from it, typing on a phone, manipulating a smartwatch, hiding it from view, or consulting it while looking at their exam paper) for a duration, then confirm 'Individual Cheating: Unauthorized Material Use'. Otherwise, consider it 'Not Cheating: Item Present but Not Used Illegally'.")

        elif event_type == 'collaborative_hand_gesture_proximity':
            p1, p2 = person_ids[0], person_ids[1] if len(person_ids) > 1 else 'another student'
            return (f"{base_prompt}Initial event: Close hand proximity detected between {p1} and {p2}, suggesting a physical interaction.\n"
                    "Constraint: If the video shows a clear visual exchange of a small object or written material between their hands, OR if one student points or makes a distinct non-verbal signal towards the other's paper, OR if one student passes written material to the other, then confirm 'Collaborative Cheating: Material Exchange/Signaling'. Otherwise, consider it 'Not Cheating: Accidental Contact'.")

        elif event_type == 'individual_suspicious_gaze':
            p_id = person_ids[0] if person_ids else 'a student'
            return (f"{base_prompt}Initial event: {p_id} exhibiting extreme or sustained gaze away from their exam paper/monitor (e.g., looking at the ceiling, floor, or consistently away from their designated area).\n"
                    "Constraint: If the video shows {p_id} looking intently or for an extended period at an area not relevant to the exam (e.g., a wall, under a desk, or another student's general direction without direct paper view) or appearing to consult hidden notes/devices, then confirm 'Individual Cheating: Suspicious Gaze/Consulting'. Otherwise, consider it 'Not Cheating: Normal Movement/Thinking'.")

        else:
            return f"{base_prompt}Initial event: {event_data.get('description', 'An unclassified suspicious event')}. Constraint: Carefully analyze the student's actions and surroundings in the provided video frames for any unusual behavior or violation of exam rules, and provide a clear decision and explanation."


