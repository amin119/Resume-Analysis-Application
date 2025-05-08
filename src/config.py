from pathlib import Path 

BASE8DIR = Path(__file__).parent.parent

#data paths 
RAW_DATA_PATH = BASE_DIR / 'data/raw/resume_dataset.csv'
PROCESSED_DATA_PATH = BASE_DIR / 'data/processed/resume_dataset.csv'
#model paths

MODEL_PATH = BASE_DIR / 'models/resume_bert_finetuned'

#RAG paths

KNOWLEDGE_BASE_DIR = BASE_DIR / 'knowledge_base'
FAISS_INDEX_DIR = BASE_DIR / 'faiss_index'
