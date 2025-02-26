# agent_manager.py
import asyncio
from agents.report_agent import ReportGenerationAgent
from agents.receipt_agent import ReceiptAnalysisAgent
from agents.diagram_agent import DiagramAnalysisAgent
from agents.training_agent import TrainingAssessmentAgent
from chatbot import ChatGPTStyleChatbot, load_config

class AgentManager:
    def __init__(self):
        self.config = load_config()
        self.chatbot = ChatGPTStyleChatbot(self.config)
        self.report_agent = ReportGenerationAgent()
        self.receipt_agent = ReceiptAnalysisAgent()
        self.diagram_agent = DiagramAnalysisAgent()
        self.training_agent = TrainingAssessmentAgent()
    
    async def process_interactive_query(self, query: str) -> str:
        query_lower = query.lower()
        if "receipt" in query_lower:
            return "Please use the receipt upload endpoint to process a receipt image."
        elif "report" in query_lower:
            return await self.report_agent.generate_report(query)
        elif "diagram" in query_lower or "troubleshoot" in query_lower:
            return await self.diagram_agent.analyze_diagram(b"")  # In real usage, pass image bytes.
        elif "quiz" in query_lower or "test" in query_lower:
            return await self.training_agent.generate_quiz(query)
        else:
            # Use the full chatbot with conversation memory and chain-of-thought.
            return await self.chatbot.complete_query(query)

if __name__ == "__main__":
    async def test():
        manager = AgentManager()
        result = await manager.process_interactive_query("Generate a report on substation safety measures")
        print(result)
    asyncio.run(test())