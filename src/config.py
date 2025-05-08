from pathlib import Path 
import os

class Config:
    BASE_DIR = Path(__file__).parent.parent
    
    # Data paths
    RAW_DATA = BASE_DIR / "data/raw/resumes.csv"
    PROCESSED_DATA = BASE_DIR / "data/processed/resumes_processed.pkl"
    
    # Model paths
    MODEL_DIR = BASE_DIR / "models/resume_bert"
    TOKENIZER_DIR = BASE_DIR / "models/tokenizer"
    
    # RAG paths
    KNOWLEDGE_BASE = BASE_DIR / "knowledge_base"
    FAISS_INDEX = BASE_DIR / "faiss_index"
    
    @classmethod
    def setup_dirs(cls):
        os.makedirs(cls.BASE_DIR / "data/processed", exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.TOKENIZER_DIR, exist_ok=True)
        os.makedirs(cls.KNOWLEDGE_BASE, exist_ok=True)