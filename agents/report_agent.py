# agents/report_agent.py
import asyncio
import logging
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger(__name__)

class ReportGenerationAgent:
    def __init__(self):
        pass

    async def generate_report(self, parameters: str) -> str:
        prompt = (
            f"Generate a detailed electrical report for the following parameters: {parameters}\n\n"
            "Include technical specifications, troubleshooting steps, and recommendations. "
            "Provide the report in a structured format with headings."
        )
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: openai.Completion.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=500,
                temperature=0.7,
            ))
            report = response.choices[0].text.strip()
            logger.info("Report generation successful.")
            return report
        except Exception as e:
            logger.error(f"Error generating report: {e}", exc_info=True)
            return "An error occurred while generating the report."