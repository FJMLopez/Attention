# src/analysis/evaluator.py

import logging
import re
from typing import List, Dict, Tuple, Optional, Any

from src.types import (
    CorpusSentence, 
    SystemSequence, 
    AnalysisResult, 
    AppConfig
)
from src.utils.alignment import compute_edit_distance

logger = logging.getLogger(__name__)

# Type alias for the tuple returned by find_coref_matches:
# (system_token_index, set_id, is_exact_match_bool)
MatchTuple = Tuple[int, str, bool]

# Type alias for grouped mentions: Dict[set_id, List[List[token_indices]]]
# Example: {'set_1': [[0, 1], [5]]} -> "The cat" ... "it"
MentionGroup = Dict[str, List[List[int]]]


def _parse_annotated_token(token: str) -> Optional[str]:
    """
    Parses a token string to extract the Coreference Set ID.
    
    Format expected: "#[word]#-set_ID"
    
    Args:
        token (str): The annotated token from CorpusSentence.
        
    Returns:
        Optional[str]: The set_id (e.g., "set_10") if found, else None.
    """
    if token.startswith('#[') and ']#-set' in token:
        # Extract the part after ']#-'
        try:
            return token.split(']#-')[-1]
        except IndexError:
            return None
    return None


def find_coref_matches(
        ref_sent: CorpusSentence, 
        sys_seq: str, 
        align_ops: List[Tuple[str, int, int]]
    ) -> List[MatchTuple]:
    """
    Identifies tokens in the System Output that correspond to Mention tokens 
    in the Reference Corpus using the computed Edit Distance alignment.

    Args:
        ref_sent (CorpusSentence): The reference sentence object containing 
                                   annotated text (with #[word]# tags).
        sys_seq (str): The raw system output string.
        align_ops (list): The alignment operations from edit_distance.
                          Format: (op_name, ref_idx, sys_idx)

    Returns:
        List[MatchTuple]: A list of tuples (sys_idx, set_id, is_exact_match).
    """
    sys_coref_idxs: List[MatchTuple] = []
    
    # We need the split lists to check for content equality
    ref_tokens_clean = ref_sent.tokenized_text.split()
    ref_tokens_annot = ref_sent.annotated_text.split()
    sys_tokens = sys_seq.split()
    
    logger.debug(f"Ref Tokens: {len(ref_tokens_annot)}")
    for op in align_ops:
        # op is (operation_name, ref_index, sys_index)
        # Note: ref_index or sys_index can be None (for insertions/deletions)
        logger.debug(f"Alignment Op: {op}")


        ref_idx = op[1]
        sys_idx = op[2]

        # We look for 'match' or 'sub' where both indices exist
        if ref_idx is not None and sys_idx is not None:
            
            # Check if the reference token marks a mention
            annot_token = ref_tokens_annot[ref_idx]
            set_id = _parse_annotated_token(annot_token)
            
            if set_id and sys_tokens[sys_idx] != '<pad>':
                # Check if the words are identical (True) or just aligned (False)
                is_exact_match = (ref_tokens_clean[ref_idx] == sys_tokens[sys_idx])
                
                sys_coref_idxs.append((sys_idx, set_id, is_exact_match))

    return sys_coref_idxs


def _group_mentions(matches: List[MatchTuple]) -> MentionGroup:
    """
    Groups individual token matches into full mention spans.
    
    Logic:
        - If two tokens have the same set_id and are adjacent indices, 
          they belong to the same mention.
        - Otherwise, they start a new mention.
    
    Args:
        matches (List[MatchTuple]): Output from find_coref_matches.

    Returns:
        MentionGroup: Dictionary mapping Set IDs to lists of token indices.
    """
    if not matches:
        return {}

    entities: MentionGroup = {}
    
    # 1. Initialize with the first match
    curr_idx = matches[0][0]
    curr_set = matches[0][1]
    
    current_span = [curr_idx]
    
    # Helper to commit a span to the dict
    def add_span(s_id, span):
        if s_id not in entities:
            entities[s_id] = []
        entities[s_id].append(span)

    # 2. Iterate through the rest
    for i in range(1, len(matches)):
        idx, set_id, _ = matches[i]
        prev_idx, prev_set, _ = matches[i-1]

        # Check continuity: Same Set ID AND Adjacent Index
        if set_id == prev_set and idx == prev_idx + 1:
            current_span.append(idx)
        else:
            # Commit the previous span
            add_span(prev_set, current_span)
            # Start new span
            current_span = [idx]
        
        curr_set = set_id

    # 3. Commit the final span
    add_span(curr_set, current_span)

    return entities


def _compute_link_scores(
        cur_mentions: MentionGroup,
        ctx_mentions: MentionGroup,
        attention_matrix: List[List[float]],
        score_mode: str = 'max'
    ) -> List[List[Any]]:
    """
    Computes attention metrics for every link between Current Sentence Mentions
    and Context Sentence Mentions.

    Metrics vector format per link:
    [
        Bool: Is the max weight of the current token pointing to the antecedent?,
        Bool: Is the antecedent weight > 0?,
        Bool: (Legacy/Unused) Placeholder,
        Float: The actual score (Max or Avg weight)
    ]

    Args:
        cur_mentions: Mentions in the hypothesis (current) sentence.
        ctx_mentions: Mentions in the context sentence.
        attention_matrix: Matrix [curr_len x ctx_len].
        score_mode: 'max' or 'avg'.

    Returns:
        List of metrics vectors.
    """
    metrics = []

    # Iterate over all Set IDs found in the Current Sentence
    for set_id in cur_mentions:
        # If this entity also exists in the Context Sentence
        if set_id in ctx_mentions:
            
            # For every specific mention span of this entity in Current
            for cur_span in cur_mentions[set_id]:
                
                # For every specific mention span of this entity in Context
                for ctx_span in ctx_mentions[set_id]:
                    
                    # Initialize Metric Vector: 
                    # [MaxIsAntecedent, HasNonZeroWeight, Unused, Score]
                    link_metric = [False, False, False, 0.0]
                    
                    all_weights_in_link = []
                    max_weight_in_rows = 0.0

                    # Analyze attention from Current Span tokens
                    for i in cur_span:
                        if i >= len(attention_matrix):
                            continue # Safety check
                            
                        # Find the max attention this token pays to ANY token in context
                        row_max = max(attention_matrix[i])
                        if row_max > max_weight_in_rows:
                            max_weight_in_rows = row_max
                        
                        # Collect weights specifically pointing to the Antecedent Span
                        for j in ctx_span:
                            if j < len(attention_matrix[i]):
                                w = attention_matrix[i][j]
                                all_weights_in_link.append(w)

                    # Calculate Score
                    if not all_weights_in_link:
                        current_score = 0.0
                    elif score_mode == 'avg':
                        current_score = sum(all_weights_in_link) / len(all_weights_in_link)
                    else:
                        current_score = max(all_weights_in_link)

                    # Metric 1: Max Weight is Antecedent
                    # True if the link score is >= the max weight the token put anywhere else
                    # AND the link actually has weight
                    if current_score >= max_weight_in_rows and sum(all_weights_in_link) > 0.0:
                        link_metric[0] = True
                    
                    # Metric 2: Has Non-Zero Weight
                    link_metric[1] = (sum(all_weights_in_link) > 0.0)
                    
                    # Metric 4: Score
                    link_metric[3] = current_score

                    metrics.append(link_metric)
    
    return metrics

def evaluate_attention(
    aligned_data: List[CorpusSentence],
    system_data: Dict[str, SystemSequence],
    config: AppConfig,
    ctx_window_size: int = 3
) -> List[AnalysisResult]:
    """
    Main evaluation loop.
    
    Iterates through the aligned corpus. For each sentence, it retrieves the 
    corresponding system output and attention matrix. It then looks back at 
    'ctx_window_size' previous sentences to find coreference links and 
    evaluate if the attention mechanism focused on the correct antecedent.

    Args:
        aligned_data (List[CorpusSentence]): The list of sentences from Corpus,
                                             matched to NMT order.
        system_data (Dict[str, SystemSequence]): The loaded system outputs.
        config (AppConfig): Configuration object (thresholds, modes).
        ctx_window_size (int): Number of context sentences to look back.

    Returns:
        List[AnalysisResult]: List containing detailed metrics for each sequence.
    """
    results: List[AnalysisResult] = []
    
    # Index determination for system data keys
    # 'concat' systems usually index from 0-0, 'multienc' from 0-1
    cur_bogus_idx = 0 if config.canmt_system == 'concat' else 1
    
    # We iterate through the aligned corpus
    for idx, ref_sent in enumerate(aligned_data):
        
        # Construct Key: "{sentence_index}-{bogus_index}"
        # Note: The original script had an 'offset' logic for specific subsets.
        # We assume here that 'aligned_data' indices align with system_data keys directly.
        # If strict offset logic is needed, it should be injected here.
        key = f"{idx}-{cur_bogus_idx}"
        
        if key not in system_data:
            # This might happen if system data is incomplete or index mismatch
            logger.debug(f"Skipping index {idx}: Key {key} not found in system data.")
            continue
            
        sys_seq_obj = system_data[key]
        cur_sys_text = sys_seq_obj.current

        # 1. Validate Quality (WER check)
        # We verify that the text in the Attention file matches the Reference text
        # closely enough to transfer annotations.
        er_res = compute_edit_distance(ref_sent.tokenized_text, cur_sys_text)
        # er_res: (ins, del, sub, len, alignment)
        wer = sum(er_res[:3]) / er_res[3] if er_res[3] > 0 else 0.0
        
        if wer >= config.wer_threshold:
            logger.error(f"FATAL: High divergence (WER {wer:.2f}) at {key}.")
            logger.error(f"Ref: {ref_sent.tokenized_text}")
            logger.error(f"Sys: {cur_sys_text}")
            continue

        alignment_ops = er_res[4]

        # 2. Get Matches in Current Sentence
        cur_matches = find_coref_matches(ref_sent, cur_sys_text, alignment_ops)
        cur_mentions_grouped = _group_mentions(cur_matches)

        # 3. Look at Context Sentences
        all_metrics_for_seq = []
        
        # Loop backwards from 1 to 3 (or whatever window size)
        for ctx_idx in range(1, ctx_window_size + 1):
            ctx_key = f"{idx}-{ctx_idx}"
            
            # Check if context exists
            if ctx_key not in system_data:
                continue
                
            ctx_sys_obj = system_data[ctx_key]
            
            # Check if we have a valid previous sentence in the corpus
            # (e.g. at index 0, we have no context)
            if idx - ctx_idx < 0:
                continue

            ref_ctx_sent = aligned_data[idx - ctx_idx]
            
            # Validate Context Quality
            ctx_er_res = compute_edit_distance(
                ref_ctx_sent.tokenized_text, 
                ctx_sys_obj.context
            )
            ctx_alignment_ops = ctx_er_res[4]
            
            # Get Matches in Context Sentence
            ctx_matches = find_coref_matches(
                ref_ctx_sent, 
                ctx_sys_obj.context, 
                ctx_alignment_ops
            )
            ctx_mentions_grouped = _group_mentions(ctx_matches)
            
            # 4. Compute Metrics using the Attention Matrix
            # Note: The matrix in system_data corresponds to the specific context sentence
            # loaded in ctx_sys_obj?
            # Actually, typically attention files contain one matrix per line 
            # representing Current->Context weights.
            # We assume sys_seq_obj.attention[ctx_idx-1] or similar logic 
            # depending on how system_loader.py parses the tabs.
            # Based on legacy: system_data has one entry per line.
            # Legacy logic implies: system_data[key]['att'] is the matrix.
            # BUT the key changes for context.
            # Let's look at legacy: `system_data[key]` retrieves the line.
            # That line contains `cur` and `ctx`.
            # So `system_data[key]['att']` is the attention between `cur` and THAT `ctx`.
            
            link_metrics = _compute_link_scores(
                cur_mentions_grouped,
                ctx_mentions_grouped,
                ctx_sys_obj.attention,
                score_mode=config.coref_link_score_mode
            )
            
            all_metrics_for_seq.extend(link_metrics)

        # 5. Store Result
        if all_metrics_for_seq:
            result = AnalysisResult(
                seq_id=key,
                current_sent=cur_sys_text,
                context_sent="[Multiple Contexts]", 
                metrics=all_metrics_for_seq
            )
            results.append(result)

    return results

