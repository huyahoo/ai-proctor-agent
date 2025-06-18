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
        """Constructs the prompt for the LLM based on event data."""
        base_prompt = (
            "You are an expert AI proctoring assistant. Your task is to generate a single, concise, and actionable instruction for a subordinate Vision-Language Model (VLM) to monitor provided suspicious actions of students. "
            f"This instruction will guide the VLM in monitoring a sequence of images representing a cut video clip of {self.config.VLM_ANALYSIS_CLIP_SECONDS} seconds."
            "that immediately follows a potential academic integrity violation during an in-person exam. Each seconds will take 2 frames.\n\n"
            f"The instruction must be a direct command to the VLM, focusing on specific, observable actions based on the suspicious event to confirm or deny cheating. "
            "Frame your response as a direct order to the VLM. Be brief, clear, and focus only on verifiable visual evidence.\n\n"
            "---"
        )
        event_description = event_data.get('description', 'An unclassified suspicious event occurred.')
        final_prompt = (
            f"{base_prompt}\n"
            f"Suspicious Event: {event_description}\n\n"
            "Instruction for VLM:"
        )
        return final_prompt


