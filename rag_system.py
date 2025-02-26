#!/usr/bin/env python
"""
Advanced RAG System for Indexing PDFs into Pinecone
"""

import os
import sys
import json
import time
import math
import hashlib
import logging
import argparse
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone  # LangChain's Pinecone wrapper
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import random

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
console = Console()

PROCESSED_FILES_PATH = "processed_files.json"

class RAGIndexer:
    def __init__(self, 
                 data_dir: Path,
                 index_name: str,
                 namespace: str,
                 pinecone_api_key: str,
                 openai_api_key: str,
                 pinecone_region: str = "us-west1-gcp",
                 embedding_dim: int = 1536,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 batch_size: int = 100,
                 reprocess: bool = False,
                 max_retries: int = 5,
                 backoff_factor: float = 1.5) -> None:
        self.data_dir = data_dir
        self.index_name = index_name
        self.namespace = namespace
        self.pinecone_api_key = pinecone_api_key
        self.openai_api_key = openai_api_key
        self.pinecone_region = pinecone_region
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.reprocess = reprocess
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.processed_files: Dict[str, str] = self.load_processed_files()
        self.index_name = self.initialize_pinecone()

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def load_processed_files(self) -> Dict[str, str]:
        processed_path = Path(PROCESSED_FILES_PATH)
        if processed_path.exists():
            try:
                with processed_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.exception(f"Error loading {PROCESSED_FILES_PATH}: {e}")
                return {}
        return {}

    def save_processed_files(self) -> None:
        try:
            with Path(PROCESSED_FILES_PATH).open("w", encoding="utf-8") as f:
                json.dump(self.processed_files, f, indent=4)
        except Exception as e:
            logger.exception(f"Error saving processed files: {e}")

    def initialize_pinecone(self) -> str:
        try:
            pc = PineconeClient(api_key=self.pinecone_api_key)
        except Exception as e:
            logger.exception(f"Error initializing Pinecone client: {e}")
            sys.exit(1)
        try:
            existing_indexes = [idx["name"] for idx in pc.list_indexes()]
        except Exception as e:
            logger.exception(f"Error listing Pinecone indexes: {e}")
            sys.exit(1)
        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index '{self.index_name}'...")
            try:
                pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=self.pinecone_region)
                )
            except Exception as e:
                logger.exception(f"Error creating Pinecone index: {e}")
                sys.exit(1)
        else:
            logger.info(f"Using existing Pinecone index '{self.index_name}'...")
        return self.index_name

    def load_and_chunk_documents(self) -> List[Dict[str, Any]]:
        pdf_files = list(self.data_dir.glob("*.pdf"))
        documents: List[Any] = []
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
            task = progress.add_task("Processing PDFs...", total=len(pdf_files))
            for pdf_file in pdf_files:
                try:
                    file_hash = self.compute_file_hash(pdf_file)
                except Exception:
                    progress.advance(task)
                    continue
                file_str = str(pdf_file.resolve())
                if (not self.reprocess and 
                    file_str in self.processed_files and 
                    self.processed_files[file_str] == file_hash):
                    logger.info(f"Skipping already processed file: {pdf_file.name}")
                    progress.advance(task)
                    continue
                logger.info(f"Processing new/updated file: {pdf_file.name}")
                try:
                    loader = PyPDFLoader(str(pdf_file))
                    docs = loader.load()
                    for doc in docs:
                        if not hasattr(doc, "metadata") or not isinstance(doc.metadata, dict):
                            doc.metadata = {}
                        doc.metadata["source_file"] = pdf_file.name
                        doc.metadata["processed_at"] = datetime.now().isoformat()
                    documents.extend(docs)
                except Exception as e:
                    logger.exception(f"Error loading {pdf_file.name}: {e}")
                    progress.advance(task)
                    continue
                self.processed_files[file_str] = file_hash
                progress.advance(task)
        self.save_processed_files()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        try:
            chunks = text_splitter.split_documents(documents)
        except Exception as e:
            logger.exception(f"Error splitting documents into chunks: {e}")
            return []
        logger.info(f"Processed {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    async def embed_and_store_documents(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            logger.info("No new documents to process. Skipping embedding step.")
            return
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        except Exception as e:
            logger.exception(f"Error initializing embeddings: {e}")
            return
        batches = [chunks[i:i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
        logger.info(f"Uploading {len(chunks)} chunks in {len(batches)} batches.")
        def process_batch(batch: List[Dict[str, Any]]) -> bool:
            attempt = 0
            delay = 1.0
            while attempt < self.max_retries:
                try:
                    Pinecone.from_documents(batch, embeddings, index_name=self.index_name, namespace=self.namespace)
                    return True
                except Exception as e:
                    attempt += 1
                    logger.exception(f"Batch upsert failed on attempt {attempt}: {e}")
                    time.sleep(delay)
                    delay *= self.backoff_factor * (1 + random.random() * 0.5)
            return False
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            tasks = [
                loop.run_in_executor(executor, partial(process_batch, batch))
                for batch in batches
            ]
            results = await asyncio.gather(*tasks)
        logger.info(f"Batch upload results: {results}")

    async def run(self) -> None:
        chunks = self.load_and_chunk_documents()
        await self.embed_and_store_documents(chunks)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced RAG System for Indexing PDFs into Pinecone")
    parser.add_argument("--data_dir", type=str, default="data/manuals", help="Directory containing PDF files")
    parser.add_argument("--namespace", type=str, default="manuals", help="Namespace to store embeddings in Pinecone")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for text splitter")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Chunk overlap for text splitter")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for upserting embeddings")
    parser.add_argument("--reprocess", action="store_true", help="Force reprocessing of all files")
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "electrical-ai-vector-db")
    pinecone_region = os.getenv("PINECONE_REGION", "us-west1-gcp")
    if not pinecone_api_key or not openai_api_key:
        logger.error("PINECONE_API_KEY and OPENAI_API_KEY must be set in your environment.")
        sys.exit(1)
    data_directory = Path(args.data_dir)
    if not data_directory.exists():
        logger.error(f"Data directory {data_directory} does not exist.")
        sys.exit(1)
    indexer = RAGIndexer(
        data_dir=data_directory,
        index_name=index_name,
        namespace=args.namespace,
        pinecone_api_key=pinecone_api_key,
        openai_api_key=openai_api_key,
        pinecone_region=pinecone_region,
        embedding_dim=1536,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        reprocess=args.reprocess,
        max_retries=5,
        backoff_factor=1.5
    )
    asyncio.run(indexer.run())

if __name__ == "__main__":
    main()