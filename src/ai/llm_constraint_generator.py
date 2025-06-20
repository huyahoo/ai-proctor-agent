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
        """Constructs the prompt for the LLM based on event data, using few-shot examples."""
        base_prompt = (
            "You are an expert AI proctoring assistant. Your task is to generate a single, concise, and actionable instruction for a subordinate Vision-Language Model (VLM). "
            "This instruction will guide the VLM in analyzing a short video clip to verify if cheating occurred, based on an initial suspicious event. "
            "The student(s) involved are identified by their visual features.\n\n"
            "Your instruction must be a direct command to the VLM, focused on specific, observable actions. "
            "Your response must contain ONLY the instruction, with no extra text like 'Instruction for VLM:'.\n\n"
            "--- EXAMPLES ---\n\n"
            "**Suspicious Event:** Student ID 1, described as 'Gender: Man. Hair: Short black hair. Clothing: White T-shirt. Glasses: None.', "
            "was seen looking at Student ID 0, described as 'Gender: Woman. Hair: Long brown hair. Clothing: Red sweater. Glasses: Yes.'.\n"
            "**Instruction for VLM:**\n"
            "Verify if the student in the 'White T-shirt' continues to look at the 'Red sweater' student's exam paper. Track head and eye movements to determine if they are copying.\n\n"
            "---\n\n"
            "**Suspicious Event:** The wrists of Student ID 5, described as 'Gender: Man. Hair: Blond hair. Clothing: Green hoodie. Glasses: None.', "
            "are not visible, and they are looking down, not at their exam paper.\n"
            "**Instruction for VLM:**\n"
            "Closely monitor the student in the 'Green hoodie'. Determine if they are interacting with unauthorized items like a phone or notes under the desk. "
            "Watch for their hands to reappear and what they might be holding.\n\n"
            "---\n\n"
            "**Suspicious Event:** Student ID 2, described as 'Gender: Woman. Hair: Black ponytail. Clothing: Grey sweatshirt. Glasses: Yes.', "
            "has an arm extended at a suspicious angle towards another student.\n"
            "**Instruction for VLM:**\n"
            "Focus on the student in the 'Grey sweatshirt'. Observe their arm and hand movements to determine if they are passing an object to another student.\n\n"
            "--- TASK ---\n"
        )
        
        event_description = event_data.get('description', 'An unclassified suspicious event occurred.')
        
        final_prompt = (
            f"{base_prompt}\n"
            f"**Suspicious Event:** {event_description}\n"
            "**Instruction for VLM:**"
        )
        return final_prompt


