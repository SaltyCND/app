#!/usr/bin/env python
"""
Super Advanced Domain-Aware ChatGPT-Style Chatbot with Deep Reasoning, Self-Verification,
and Backend-Managed Conversation Memory

This chatbot:
- Uses a conversation memory manager that auto-summarizes if too long.
- Retrieves context from Pinecone via a RetrievalService.
- Uses an LLM (e.g., GPT-4) to generate detailed answers with chain-of-thought reasoning and self-verification.
- Provides a CLI interface with commands.
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
from typing import Optional, List, Dict

from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

# External libraries for Pinecone and LLM
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# -----------------------------------------------------------------------------
# Configuration & Logger Setup
# -----------------------------------------------------------------------------

def load_config() -> dict:
    load_dotenv()  # Load from .env file
    config = {
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "INDEX_NAME": os.getenv("PINECONE_INDEX", "electrical-ai-vector-db"),
        "NAMESPACE": os.getenv("PINECONE_NAMESPACE", "manuals"),
        "TOP_K": int(os.getenv("SIMILARITY_TOP_K", 10)),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
        "LLM_MODEL": os.getenv("LLM_MODEL", "gpt-4"),
        "SUMMARY_THRESHOLD": int(os.getenv("SUMMARY_THRESHOLD", 3500)),
    }
    missing = [k for k, v in config.items() if v is None and k in ("PINECONE_API_KEY", "OPENAI_API_KEY")]
    if missing:
        logging.error(f"Missing configuration for: {', '.join(missing)}")
        sys.exit(1)
    return config

def setup_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler("app.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

logger = setup_logger()
console = Console()

# -----------------------------------------------------------------------------
# Conversation Memory Manager with Auto-Summarization
# -----------------------------------------------------------------------------

class ConversationMemoryManager:
    def __init__(self, system_prompt: str, summary_threshold: int = 3500):
        # Start with the system prompt as the initial message.
        self.messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        self.summary_threshold = summary_threshold

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def get_history_as_text(self) -> str:
        # Concatenate conversation messages into a single text block.
        return "\n\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.messages)

    async def maybe_summarize(self, llm_service: "LLMService") -> None:
        history_text = self.get_history_as_text()
        if len(history_text) > self.summary_threshold:
            console.print("[yellow]Summarizing conversation history...[/yellow]")
            prompt = (
                "Summarize the following conversation between the user and the assistant, "
                "preserving key details and technical information:\n\n" + history_text
            )
            summary = await llm_service.complete_async(prompt)
            # Always preserve the original system prompt.
            system_msg = self.messages[0]
            self.messages = [system_msg, {"role": "system", "content": f"Summary: {summary}"}]
            logger.info("Conversation history summarized.")

# -----------------------------------------------------------------------------
# Asynchronous Retrieval Service with Caching
# -----------------------------------------------------------------------------

class RetrievalService:
    def __init__(self, config: dict):
        self.config = config
        self.pc = Pinecone(api_key=self.config["PINECONE_API_KEY"])
        if self.config["INDEX_NAME"] not in self.pc.list_indexes().names():
            logger.error(f"Pinecone index '{self.config['INDEX_NAME']}' does not exist. Create it first.")
            sys.exit(1)
        self.vector_store = PineconeVectorStore(
            self.pc.Index(self.config["INDEX_NAME"]),
            namespace=self.config["NAMESPACE"]
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        self.retriever = VectorIndexRetriever(index=self.index, similarity_top_k=self.config["TOP_K"])
        self.query_engine = RetrieverQueryEngine(retriever=self.retriever)
        self.cache: Dict[str, Optional[str]] = {}

    async def query_async(self, user_query: str) -> Optional[str]:
        if user_query in self.cache:
            logger.info("Returning cached context from Pinecone.")
            return self.cache[user_query]
        logger.info(f"Querying Pinecone for context: {user_query}")
        try:
            response = await asyncio.to_thread(self.query_engine.query, user_query)
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
            return None

        if hasattr(response, "source_nodes") and response.source_nodes:
            context = "\n\n".join(node.text for node in response.source_nodes)
            logger.info(f"Retrieved {len(response.source_nodes)} documents.")
            self.cache[user_query] = context
            return context
        else:
            logger.warning("No documents retrieved from Pinecone.")
            self.cache[user_query] = None
            return None

# -----------------------------------------------------------------------------
# Asynchronous LLM Service with Advanced Prompting
# -----------------------------------------------------------------------------

class LLMService:
    def __init__(self, config: dict):
        self.config = config
        self.embedding_model = OpenAIEmbedding(
            model=self.config["EMBEDDING_MODEL"],
            api_key=self.config["OPENAI_API_KEY"]
        )
        self.llm = OpenAI(
            model=self.config["LLM_MODEL"],
            api_key=self.config["OPENAI_API_KEY"]
        )

    async def complete_async(self, prompt: str) -> str:
        retries = 2
        for attempt in range(retries):
            try:
                response = await asyncio.to_thread(self.llm.complete, prompt)
                return response.text.strip()
            except Exception as e:
                logger.error(f"LLM error on attempt {attempt+1}: {e}")
                await asyncio.sleep(2)
        return "I'm sorry, I encountered an error processing your request."

    async def stream_complete_async(self, prompt: str) -> str:
        full_response = await self.complete_async(prompt)
        words = full_response.split()
        streamed_response = ""
        for word in words:
            streamed_response += word + " "
            console.print(word, end=" ", style="bold magenta")
            await asyncio.sleep(0.1)
        console.print()  # New line after streaming
        return full_response

# -----------------------------------------------------------------------------
# Advanced ChatGPT-Style Chatbot Class with Backend-Managed Memory
# -----------------------------------------------------------------------------

class ChatGPTStyleChatbot:
    def __init__(self, config: dict):
        system_prompt = (
            "You are a world-class electrical engineering expert specializing in substation equipment and troubleshooting. "
            "When answering questions, use a detailed, step-by-step chain-of-thought process. Verify your reasoning at each step "
            "and provide a final, comprehensive answer with clear explanations."
        )
        self.memory = ConversationMemoryManager(system_prompt, summary_threshold=config["SUMMARY_THRESHOLD"])
        self.retrieval = RetrievalService(config)
        self.llm_service = LLMService(config)
        self.config = config
        logger.info("Advanced ChatGPT-Style Chatbot initialized.")

    def build_prompt(self, user_input: str, retrieved_context: Optional[str]) -> str:
        parts = []
        for msg in self.memory.messages:
            parts.append(f"{msg['role'].capitalize()}: {msg['content']}")
        if retrieved_context:
            parts.append("System: Retrieved context:\n" + retrieved_context)
        parts.append("User: " + user_input)
        parts.append(
            "Assistant: Please think step-by-step, showing your full chain-of-thought. "
            "After reasoning, verify your solution and provide a final, comprehensive answer. "
            "If you need additional context, please ask."
        )
        return "\n\n".join(parts)

    async def refine_answer(self, prompt: str, initial_answer: str) -> str:
        refinement_prompt = (
            "Review the following answer for completeness, accuracy, and detail. "
            "If improvements are needed, provide a revised final answer with thorough explanations.\n\n"
            "Initial Answer: " + initial_answer + "\n\nPrompt: " + prompt
        )
        refined_answer = await self.llm_service.complete_async(refinement_prompt)
        return refined_answer if refined_answer.strip() != "" else initial_answer

    async def complete_query(self, query: str) -> str:
        # Retrieve domain-specific context.
        retrieved_context = await self.retrieval.query_async(query)
        # Build the prompt.
        prompt = self.build_prompt(query, retrieved_context)
        # Optionally summarize conversation history if it exceeds the threshold.
        await self.memory.maybe_summarize(self.llm_service)
        # Get initial answer.
        initial_answer = await self.llm_service.complete_async(prompt)
        # Refine the answer.
        final_answer = await self.refine_answer(prompt, initial_answer)
        # Update conversation history.
        self.memory.add_message("user", query)
        self.memory.add_message("assistant", final_answer)
        return final_answer.strip()

    async def generateAnswer(self, query: str) -> str:
        """
        Generates the final answer without updating conversation memory.
        Used exclusively by streaming endpoints.
        """
        retrieved_context = await self.retrieval.query_async(query)
        prompt = self.build_prompt(query, retrieved_context)
        await self.memory.maybe_summarize(self.llm_service)
        initial_answer = await self.llm_service.complete_async(prompt)
        final_answer = await self.refine_answer(prompt, initial_answer)
        return final_answer.strip()

    async def chat_loop(self) -> None:
        console.print("\n[bold green]🚀 Welcome to the Advanced ChatGPT-Style Chatbot![/bold green]")
        console.print("Type [cyan]'help'[/cyan] for commands, [cyan]'exit'[/cyan] to quit.\n")
        while True:
            try:
                user_input = Prompt.ask("[bold blue]You[/bold blue]").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[bold green]👋 Goodbye![/bold green]")
                break

            if user_input.lower() in ["exit", "quit"]:
                console.print("\n[bold green]👋 Goodbye![/bold green]")
                break
            if user_input.lower() == "clear":
                system_msg = self.memory.messages[0]
                self.memory = ConversationMemoryManager(system_msg["content"], summary_threshold=self.config["SUMMARY_THRESHOLD"])
                console.print("[yellow]Conversation history cleared.[/yellow]")
                continue
            if user_input.lower() == "export":
                await self.export_conversation()
                continue
            if user_input.lower() == "help":
                self.display_help()
                continue

            retrieved_context = await self.retrieval.query_async(user_input)
            prompt = self.build_prompt(user_input, retrieved_context)
            await self.memory.maybe_summarize(self.llm_service)
            console.print("\n[bold magenta]🤖 Chatbot (Initial Answer):[/bold magenta] ", end="")
            initial_answer = await self.llm_service.stream_complete_async(prompt)
            final_answer = await self.refine_answer(prompt, initial_answer)
            self.memory.add_message("user", user_input)
            self.memory.add_message("assistant", final_answer)
            console.print(f"\n[bold magenta]🤖 Chatbot Final Answer:[/bold magenta] {final_answer}\n")
            console.print("\n[bold yellow]Conversation History:[/bold yellow]")
            console.print(self.memory.get_history_as_text())
            console.print("\n" + "=" * 50 + "\n")

    async def export_conversation(self, filename: str = "conversation_export.json") -> None:
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.memory.messages, f, indent=2)
            console.print(f"[green]Conversation exported to {filename}[/green]")
        except Exception as e:
            logger.error(f"Error exporting conversation: {e}")
            console.print("[red]Failed to export conversation.[/red]")

    def display_help(self) -> None:
        table = Table(title="Chatbot Commands")
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description", style="magenta")
        table.add_row("exit", "Exit the chatbot")
        table.add_row("clear", "Clear conversation history (except system prompt)")
        table.add_row("export", "Export conversation history to a file")
        table.add_row("help", "Display this help message")
        console.print(table)

# -----------------------------------------------------------------------------
# Main CLI Entry Point
# -----------------------------------------------------------------------------

async def main_chatbot():
    parser = argparse.ArgumentParser(description="Super Advanced Domain-Aware Chatbot")
    parser.add_argument("--top_k", type=int, default=10, help="Number of documents to retrieve")
    parser.add_argument("--summary_threshold", type=int, default=3500, help="Character threshold for summarization")
    args = parser.parse_args()
    config = load_config()
    config["TOP_K"] = args.top_k
    config["SUMMARY_THRESHOLD"] = args.summary_threshold
    chatbot = ChatGPTStyleChatbot(config)
    try:
        await chatbot.chat_loop()
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main_chatbot())