# src/data/system_loader.py
from pathlib import Path
import logging
from src.types import SystemSequence

logger = logging.getLogger(__name__)

def load_system_data(filepath: Path) -> dict[str, SystemSequence]:
    """
    Reads the system output file containing attention weights.
    Structure: 
    Line 1: file_path
        Inside file_path:
        Line 1: context_seq (tab separated)
        Line 2..N: target_token \t weight_1 \t weight_2 ...
    """
    system_data = {}
    
    with open(filepath, encoding='utf-8') as f:
        file_list = [l.strip() for l in f.readlines()]

    for subfile in file_list:
        path = Path(subfile)
        if not path.exists():
            logger.warning(f"Attention file not found: {path}")
            continue

        with open(path, encoding='utf-8') as sf:
            lines = sf.readlines()
            
        if not lines:
            continue

        # Header: seq_id \t context_tokens...
        header_parts = lines[0].strip().split('\t')
        seq_id = header_parts[0]
        ctx_seq = ' '.join(header_parts[1:])
        
        cur_tokens = []
        attention_matrix = []
        
        for line in lines[1:]:
            parts = line.strip().split('\t')
            cur_tokens.append(parts[0])
            # Clean weights (handle '.' as 0.0)
            weights = [float(w) if w != '.' else 0.0 for w in parts[1:]]
            attention_matrix.append(weights)

        cur_seq = ' '.join(cur_tokens)
        system_data[seq_id] = SystemSequence(cur_seq, ctx_seq, attention_matrix)

    return system_data