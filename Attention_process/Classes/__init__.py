# attention_lib/__init__.py
from .Sentence import Sentence
from .utils import get_documents_heads
from .config import HEAD_DOCUMENTS_FILE

__all__ = ["Sentence", "get_documents_heads", "HEAD_DOCUMENTS_FILE"]