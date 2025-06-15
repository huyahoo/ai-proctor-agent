import google.generativeai as genai
from core.config import Config

class LLMConstraintGenerator:
    def __init__(self, config: Config):
        self.config = config
        genai.configure(api_key=self.config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(self.config.LLM_MODEL_NAME)
        print(f"LLM model '{self.config.LLM_MODEL_NAME}' initialized for constraint generation.")

    def generate_constraints(self, event_data: dict) -> str:
        """
        Generates dynamic constraints for the VLM based on a detected anomaly event.
        Args:
            event_data (dict): A dictionary describing the initial anomaly event.
                               Example: {'type': 'collaborative_gaze_correlation', 'person_ids': ['Person_0', 'Person_1'], 'timestamp': 12.3}
        Returns:
            str: A natural language string of constraints for the VLM.
        """
        prompt = self._build_prompt(event_data)
        try:
            response = self.model.generate_content(prompt)
            # Ensure the response structure is as expected
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                print("LLM response did not contain expected content.")
                return "Analyze for further suspicious activity."
        except Exception as e:
            print(f"Error calling LLM for constraint generation: {e}")
            return "Analyze for any further suspicious activity."

    def _build_prompt(self, event_data: dict) -> str:
        """Constructs the prompt for the LLM based on event data."""
        event_type = event_data.get('type', 'unknown')
        person_ids = event_data.get('person_ids', [])
        object_label = event_data.get('object_label', '')

        base_prompt = "An initial suspicious event has been detected during an exam. Based on the event type, generate a concise, specific instruction or a set of 'if-then' constraints for a vision-language model (VLM) to verify if actual cheating is occurring. Focus on observable actions in the next few seconds.\n\n"

        if event_type == 'collaborative_gaze_correlation':
            p1, p2 = person_ids[0], person_ids[1] if len(person_ids) > 1 else 'another student'
            return (f"{base_prompt}Initial event: Correlated gaze detected between {p1} and {p2}.\n"
                    "Constraint: If either student's eyes remain fixed on the other's paper for more than 2 seconds, OR if either student gestures towards the other's paper, OR if any object is exchanged, then confirm 'Collaborative Cheating: Peeking/Signaling'. Otherwise, consider it 'Not Cheating: Accidental Glance'.")

        elif event_type == 'individual_unauthorized_material':
            p_id = person_ids[0] if person_ids else 'a student'
            return (f"{base_prompt}Initial event: {p_id} detected with a potential unauthorized item ({object_label}).\n"
                    "Constraint: If the student interacts with the detected item (e.g., looking at it, typing on it, hiding it) OR if they consult it while looking at their exam paper, then confirm 'Individual Cheating: Unauthorized Material Use'. Otherwise, consider it 'Not Cheating: Item Present but Not Used Illegally'.")

        elif event_type == 'collaborative_hand_gesture_proximity':
            p1, p2 = person_ids[0], person_ids[1] if len(person_ids) > 1 else 'another student'
            return (f"{base_prompt}Initial event: Close hand proximity detected between {p1} and {p2}.\n"
                    "Constraint: If a small object is visually exchanged between their hands, OR if one student points or makes a distinct signal towards the other, OR if one student passes written material to the other, then confirm 'Collaborative Cheating: Material Exchange/Signaling'. Otherwise, consider it 'Not Cheating: Accidental Contact'.")

        else:
            return f"{base_prompt}Initial event: {event_data.get('description', 'An unknown suspicious event')}.\nConstraint: Carefully analyze the student's actions and surroundings for any unusual behavior or violation of exam rules."


