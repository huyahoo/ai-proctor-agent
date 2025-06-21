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
            max_output_tokens=512,
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

        logger.info(f"\nLLM Prompt: {prompt}")
        
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
                return "Monitoring for any further suspicious activity in the video segment."
        except Exception as e:
            logger.error(f"Error calling LLM for constraint generation: {e}")
            return f"Monitoring for any further suspicious activity. LLM error: {e}"

    def _build_prompt(self, event_data: dict) -> str:
        """
        Constructs a dynamic and specific prompt for the LLM based on the event type.
        This function creates a high-quality prompt to guide the LLM in generating a
        precise and actionable instruction for the VLM.
        """
        event_type = event_data.get('type', 'unknown')
        event_description = event_data.get('description', 'An unclassified suspicious event occurred.')

        # Define the specific verification goal for each anomaly type.
        if event_type == 'suspicious_arm_angle':
            instruction_goal = """
                Determine if the student is holding unauthorized object or passing an object or signaling to another student. Focus on their arm and hand movements to see if an object is exchanged.
            """
        elif event_type == 'mutual_gaze':
            instruction_goal = """
                Determine if the students are communicating. Analyze their mouth movements for talking, facial expressions for non-verbal cues, and any subtle hand gestures between them. Or do they have potentially exchange unauthorized object.
            """
        elif event_type == 'one_way_gaze':
            instruction_goal = """
                Verify if the student initiating the gaze is copying from the other student's paper. Track their head and eye movements for a pattern of looking at the other paper and then immediately writing on their own.
            """
        elif event_type == 'suspicious_under_table':
            instruction_goal = """
                Determine if the student is using an unauthorized item (e.g., phone, notes). Monitor the area under their desk and watch for their hands to reappear, identifying anything they might be holding.
            """
        elif event_type == 'copying_others_answer':
            instruction_goal = """
                Confirm if the student is actively copying from a neighbor. Cross-reference their gaze and head movements with their writing actions, looking for a clear pattern of glancing at another student's paper and then writing.
            """
        else:
            instruction_goal = """
                Monitor the student for any general suspicious activity that could indicate cheating.
            """

        # This detailed base prompt uses a one-shot example to teach the LLM its task.
        base_prompt = f"""You are an expert AI proctoring assistant. Your task is to generate a single, concise, and actionable instruction for a subordinate Vision-Language Model (VLM).

        --- INSTRUCTIONS ---
        1. You will receive a "Suspicious Event Description" detailing an anomaly and the student(s) involved.
        2. You will also receive an "Instruction Goal" outlining what the VLM needs to verify.
        3. Your task is to synthesize this information into a single, direct command for the VLM.
        4. The command must start with "Monitoring suspicious actions of..." and must incorporate the specific student descriptions from the event to ensure the VLM knows who to watch.
        5. Your response MUST ONLY be the generated instruction for the VLM, with no extra text or markdown.

        --- EXAMPLE ---
        **Suspicious Event Description:** Student ID 1, described as 'Gender: Man. Hair: Short black hair. Clothing: White T-shirt. Glasses: None.', was seen looking at Student ID 0, described as 'Gender: Woman. Hair: Long brown hair. Clothing: Red sweater. Glasses: Yes.'.
        **Instruction Goal:** Verify if the student initiating the gaze is copying from the other student's paper. Track their head and eye movements for a pattern of looking at the other paper and then immediately writing on their own.
        **Your Generated Instruction for VLM:**
        Monitoring suspicious actions of Student ID 1 (A man in White T-shirt has short black hair). Verify if they continue to look at the student in Red sweater (A woman in Red sweater has long brown hair and wearing a round thin glasses) exam paper by tracking their head and eye movements to determine if they are copying.

        --- YOUR TASK ---
        **Suspicious Event Description:** {event_description}
        **Instruction Goal:** {instruction_goal}
        **Your Generated Instruction for VLM:**
        """

        return base_prompt


