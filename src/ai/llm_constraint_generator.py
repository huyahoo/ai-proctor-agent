import google.generativeai as genai
import re
from core.config import Config
from core.logger import logger

class LLMConstraintGenerator:
    def __init__(self, config: Config):
        self.config = config
        genai.configure(api_key=self.config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(self.config.LLM_MODEL_NAME)
        self.generation_config = genai.types.GenerationConfig(
            max_output_tokens=100,
            temperature=0.2,
        )
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
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                constraint = response.candidates[0].content.parts[0].text
                return constraint.strip()
            else:
                logger.warning("LLM response did not contain expected content or was blocked.")
                return "Analyze for any further suspicious activity in the video segment."
        except Exception as e:
            logger.error(f"Error calling LLM for constraint generation: {e}")
            return f"Analyze for any further suspicious activity. LLM error: {e}"

    def _get_event_description(self, event_data: dict) -> str:
        """Creates a natural language description of the event."""
        logger.info(f"Event data: {event_data}")
        event_type = event_data.get('type', 'unknown_event')
        person_ids = event_data.get('person_ids', [])
        object_label = event_data.get('object_label', '')

        if event_type == 'collaborative_gaze_correlation':
            p1, p2 = person_ids[0], person_ids[1] if len(person_ids) > 1 else 'another student'
            return f"Sustained, correlated side-gaze detected between students {p1} and {p2}."

        elif event_type == 'individual_unauthorized_material':
            p_id = person_ids[0] if person_ids else 'a student'
            return f"Student {p_id} detected with a potential unauthorized item ('{object_label}') in their immediate vicinity."

        elif event_type == 'collaborative_hand_gesture_proximity':
            p1, p2 = person_ids[0], person_ids[1] if len(person_ids) > 1 else 'another student'
            return f"Close hand proximity detected between students {p1} and {p2}, suggesting a physical interaction."

        elif event_type == 'individual_suspicious_gaze':
            p_id = person_ids[0] if person_ids else 'a student'
            return f"Student {p_id} is exhibiting extreme or sustained gaze away from their exam paper."

        else:
            return event_data.get('description', 'An unclassified suspicious event occurred.')

    def _build_prompt(self, event_data: dict) -> str:
        """Constructs the prompt for the LLM based on event data."""
        base_prompt = (
            "You are an expert AI proctoring assistant. Your task is to generate a single, concise, and actionable instruction for a subordinate Vision-Language Model (VLM). "
            "This instruction will guide the VLM in analyzing a 5-second video clip that immediately follows a potential academic integrity violation during an in-person exam.\n\n"
            "The instruction must be a direct command to the VLM, focusing on specific, observable actions to confirm or deny cheating. "
            "Frame your response as a direct order to the VLM. Be brief, clear, and focus only on verifiable visual evidence.\n\n"
            "---"
        )
        event_description = self._get_event_description(event_data)
        final_prompt = (
            f"{base_prompt}\n"
            f"Initial Event: {event_description}\n\n"
            "Instruction for VLM:"
        )
        return final_prompt


