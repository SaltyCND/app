# agents/training_agent.py
import asyncio
import logging
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger(__name__)

class TrainingAssessmentAgent:
    async def generate_quiz(self, topic: str) -> str:
        prompt = (
            f"Generate a quiz with 5 multiple-choice questions about the electrical topic: {topic}. "
            "Each question should have 4 options, and indicate the correct answer. "
            "Format the quiz with numbered questions and clearly mark the correct option."
        )
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: openai.Completion.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=300,
                temperature=0.7,
            ))
            quiz = response.choices[0].text.strip()
            logger.info("Quiz generation successful.")
            return quiz
        except Exception as e:
            logger.error(f"Error generating quiz: {e}", exc_info=True)
            return "Error generating the quiz."