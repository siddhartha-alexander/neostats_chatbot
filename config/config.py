# config/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(".env"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PDF_DATA_PATH = "data/researxh paper.pdf"
CHROMA_COLLECTION_NAME = "neostats"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-1.5-flash"
RAG_SCORE_THRESHOLD = 1.2
RAG_TOP_K = 2
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100