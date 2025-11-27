import sys
from typing import List, Tuple, Optional, TypeAlias

# --- DÉFINITION DES TYPES ---
# Format: (Nom Opération, Index Ref, Index Hyp)
# Utilisation de TypeAlias (Python 3.10+) pour la clarté
AlignmentOp: TypeAlias = Tuple[str, Optional[int], Optional[int]]

# Format: (n_ins, n_del, n_sub, ref_len, alignment)
EditDistanceResult: TypeAlias = Tuple[int, int, int, int, List[AlignmentOp]]

def str_edit_distance(str_ref: str, str_hyp: str) -> EditDistanceResult:
    """
    Computes Levenshtein edit distance between two strings (word-level).
    
    Args:
        str_ref: The reference string (Gold standard).
        str_hyp: The hypothesis string (System output).
        
    Returns:
        A tuple containing:
        - Number of insertions
        - Number of deletions
        - Number of substitutions
        - Length of reference
        - List of alignment operations
    """
    
    # Tokenize by splitting on whitespace
    ref = str_ref.split()
    hyp = str_hyp.split()
    
    len_reference = len(ref)
    len_hypothese = len(hyp)
    
    # Costs
    DEL_COST = 1
    INS_COST = 1
    SUB_COST = 1
    
    # Initialize Distance Matrix (N+1 x M+1)
    # d[i][j] holds the distance between ref[:i] and hyp[:j]
    distance = [[0] * (len_hypothese + 1) for _ in range(len_reference + 1)]
    
    # Initialize boundaries
    for i in range(len_reference + 1):
        distance[i][0] = i * DEL_COST
    for j in range(len_hypothese + 1):
        distance[0][j] = j * INS_COST
        
    # Forward Pass: Compute Distances
    for i in range(1, len_reference + 1):
        for j in range(1, len_hypothese + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost = 0
            else:
                cost = SUB_COST
            
            distance[i][j] = min(
                distance[i - 1][j] + DEL_COST,      # Deletion
                distance[i][j - 1] + INS_COST,      # Insertion
                distance[i - 1][j - 1] + cost       # Substitution / Match
            )

    # Backward Pass: Compute Alignment (Backtracking)
    alignment: List[AlignmentOp] = []
    n_ins = 0
    n_del = 0
    n_sub = 0
    
    i, j = len_reference, len_hypothese
    
    while i > 0 or j > 0:
        # Current costs
        curr_cost = distance[i][j]
        
        # Check operations
        # 1. Substitution or Match (Diagonal)
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else SUB_COST
            if distance[i - 1][j - 1] + cost == curr_cost:
                if cost == 0:
                    alignment.append(('match', i - 1, j - 1))
                else:
                    alignment.append(('sub', i - 1, j - 1))
                    n_sub += 1
                i -= 1
                j -= 1
                continue

        # 2. Deletion (Up) - Word in Ref but not in Hyp
        if i > 0 and distance[i - 1][j] + DEL_COST == curr_cost:
            alignment.append(('del', i - 1, None))
            n_del += 1
            i -= 1
            continue

        # 3. Insertion (Left) - Word in Hyp but not in Ref
        if j > 0 and distance[i][j - 1] + INS_COST == curr_cost:
            alignment.append(('ins', None, j - 1))
            n_ins += 1
            j -= 1
            continue

    # Reverse alignment because we backtracked from end to start
    alignment.reverse()
    
    return (n_ins, n_del, n_sub, len_reference, alignment)

def main(args: List[str]):
    if len(args) < 3:
        print("Usage: python edit_distance.py <reference_string> <hypothesis_string>")
        sys.exit(1)

    ref_str = args[1]
    hyp_str = args[2]

    print(' * Computing edit distance between:')
    print(f' * ref: {ref_str}')
    print(f' * hyp: {hyp_str}')
    print(' ---')

    n_ins, n_del, n_sub, ref_len, alignment = str_edit_distance(ref_str, hyp_str)

    wer = (n_ins + n_del + n_sub) / ref_len if ref_len > 0 else 0.0

    print(f' * ER: {wer:.2f}')
    print(' * Errors:')
    print(f' * ins: {n_ins}')
    print(f' * del: {n_del}')
    print(f' * sub: {n_sub}')
    print(' ---')

    ref_toks = ref_str.split()
    hyp_toks = hyp_str.split()
    
    print('* Alignment:')
    for op, r_idx, h_idx in alignment:
        r_word = ref_toks[r_idx] if r_idx is not None else '-'
        h_word = hyp_toks[h_idx] if h_idx is not None else '-'
        print(f' * {op}) r:{r_word}, h:{h_word}')
    print(' ---')

if __name__ == '__main__':
    main(sys.argv)