# Version of your package
__version__ = "1.0.0"

# Explicit imports for public API
from .config import Config
from .data_preprocessing import Preprocessor
from .train_model import ModelTrainer
from .rag_system import HRPolicyRetriever
from .app import app

# Optional: Package initialization code
def initialize():
    """Initialize package resources"""
    Config.setup_dirs()

__all__ = ['Config', 'Preprocessor', 'ModelTrainer', 'HRPolicyRetriever', 'app']