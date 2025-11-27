# src/utils/alignment.py
import logging
from typing import List, Tuple, Optional, TypeAlias

from src.utils.edit_distance import str_edit_distance, EditDistanceResult, AlignmentOp

logger = logging.getLogger(__name__)

def compute_edit_distance(ref: str, hyp: str) -> EditDistanceResult:
    """
    Wrapper for edit distance. 
    Returns: (ins, del, sub, total_ops, alignment_list)
    """
    return str_edit_distance(ref, hyp)
    