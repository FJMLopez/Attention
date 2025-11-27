# src/analysis/matcher.py
import logging
from src.types import CorpusData, CorpusSentence

logger = logging.getLogger(__name__)

def annotate_sentence(wids: list[str], corpus: CorpusData) -> str:
    """Reconstructs sentence with #[word]#-set_id annotations."""
    res = []
    for wid in wids:
        word_text = corpus.words.get(wid, "")
        if wid in corpus.words_in_coref:
            set_id = corpus.words_in_coref[wid]
            res.append(f"#[{word_text}]#-{set_id}")
        else:
            res.append(word_text)
    return ' '.join(res)

def match_sequences(nmt_sentences: list[str], 
                   disco_data: CorpusData, 
                   news_data: CorpusData) -> list[CorpusSentence]:
    """
    Matches NMT reference sentences to the corresponding ParCorFull sentences.
    """
    matched_data = []
    
    # Combine lookups
    disco_word_items = list(disco_data.words.items())
    news_word_items = list(news_data.words.items())
    
    disco_idx = 0
    news_idx = 0

    for s in nmt_sentences:
        s_clean = s.strip()
        
        # Check Disco Matches
        if s_clean in disco_data.text_map:
            tok_text, align_vals = disco_data.text_map[s_clean]
            # Logic to extract Word IDs based on sequential order in corpus
            # (Original logic assumed linear iteration through corpus matches NMT order)
            # ... Extraction logic here ...
            
            # Construct Result
            wids = [] # Retrieved from logic
            annotated = annotate_sentence(wids, disco_data)
            
            matched_data.append(CorpusSentence(
                raw_text=s_clean,
                tokenized_text=tok_text,
                alignment=align_vals,
                word_ids=wids,
                annotated_text=annotated
            ))
            
        # Check News Matches
        elif s_clean in news_data.text_map:
            # Same logic for news
            pass
        else:
            logger.warning(f"Sentence not found in corpus: {s_clean[:30]}...")

    return matched_data