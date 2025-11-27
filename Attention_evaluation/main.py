# main.py
import sys
import logging
from pathlib import Path

from src.config import parse_args
from src.utils.logging_setup import setup_logging
from src.utils.tokenization import Tokenizer
from src.data.parcor_loader import ParCorLoader
from src.data.system_loader import load_system_data
from src.analysis.matcher import match_sequences
from src.analysis.evaluator import evaluate_attention

def read_lines(path: Path) -> list[str]:
    with open(path, encoding='utf-8') as f:
        return [l.strip() for l in f.readlines()]

def main():    
    cfg = parse_args()
    setup_logging(verbose=cfg.verbose, debug=True, log_file=cfg.log_file)
    logger = logging.getLogger(__name__)
    logger.info("************************************************")
    logger.info(f"Starting analysis for {cfg.evaluate_language}")
    logger.info(f"Loading Corpus from {cfg.parcor_base_path}")
    logger.info("************************************************")
    # 1. Load Raw Corpus Text
    src_sentences = read_lines(cfg.corpus_source)
    logger.debug(f"Loaded {len(src_sentences)} from source corpus: {cfg.corpus_source}")
    tgt_sentences = read_lines(cfg.corpus_target)
    logger.debug(f"Loaded {len(tgt_sentences)} from target corpus: {cfg.corpus_target}")

    # 2. Load ParCorFull Data (XMLs)
    loader = ParCorLoader(cfg.parcor_base_path)
    
    logger.info("*** Loading DiscoMT data... ***")
    # Note: Logic for separate EN/DE directories needs to be handled inside loader based on lang
    disco_src = loader.load_disco_mt('EN') 
    logger.info(f"Disco_mt src loaded")
    logger.debug(f"* coref_mentions: {len(disco_src.coref_mentions)}")
    logger.debug(f"* text_map: {len(disco_src.text_map.keys())}")
    logger.debug(f"* words: {len(disco_src.words.keys())}")
    logger.debug(f"* words in coref: {len(disco_src.words_in_coref.keys())}")
    disco_tgt = loader.load_disco_mt('DE')
    logger.info(f"Disco_mt tgt loaded")
    logger.debug(f"* coref_mentions: {len(disco_tgt.coref_mentions)}")
    logger.debug(f"* text_map: {len(disco_tgt.text_map.keys())}")
    logger.debug(f"* words: {len(disco_tgt.words.keys())}")
    logger.debug(f"* words in coref: {len(disco_tgt.words_in_coref.keys())}")

    
    logger.info("*** Loading News data... ***")
    news_src = loader.load_news('EN')
    logger.info(f"News src loaded")
    logger.debug(f"* coref_mentions: {len(news_src.coref_mentions)}")
    logger.debug(f"* text_map: {len(news_src.text_map.keys())}")
    logger.debug(f"* words: {len(news_src.words.keys())}")
    logger.debug(f"* words in coref: {len(news_src.words_in_coref.keys())}")


    news_tgt = loader.load_news('DE')
    logger.info(f"*** Disco_mt tgt loaded ***")
    logger.debug(f"* coref_mentions: {len(news_tgt.coref_mentions)}")
    logger.debug(f"* text_map: {len(news_tgt.text_map.keys())}")
    logger.debug(f"* words: {len(news_tgt.words.keys())}")
    logger.debug(f"* words in coref: {len(news_tgt.words_in_coref.keys())}")


    # 3. Align NMT Input sentences to ParCor Data
    logger.info("*** Matching NMT sentences to Corpus... ***")
    aligned_src = match_sequences(src_sentences, disco_src, news_src)
    logger.info(f"Source side DONE")
    logger.debug(f"len(aligned_src): {len(aligned_src)}")
    aligned_tgt = match_sequences(tgt_sentences, disco_tgt, news_tgt)
    logger.info(f"Target side DONE")
    logger.debug(f"len(aligned_tgt): {len(aligned_tgt)}")

    # 4. Load System Attention Data
    logger.info("*** Loading System Attention Maps... ***")
    system_data = load_system_data(cfg.system_data)
    logger.info("system_data loaded.")
    logger.debug(f"system_data nb matrice: {len(system_data.keys())}")

    # 5. Run Analysis
    logger.info("Evaluating Attention...")
    target_alignment = aligned_src if cfg.evaluate_language == 'source' else aligned_tgt
    logger.debug(f"Number of sequences to evaluate: {len(target_alignment)}")
    logger.debug(f"len src aligned: {len(target_alignment)}")
    logger.debug(f"src aligned: {target_alignment}")

    results = evaluate_attention(target_alignment, system_data, cfg)

    # 6. Output
    logger.info(f"Writing results to {cfg.output_file}")
    with open(cfg.output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(f"{res.seq_id}\n")
            f.write(f"current: {res.current_sent}\n")
            f.write(f"context: {res.context_sent}\n")
            f.write(" -----\n")

    logger.info("Analysis Complete.")

if __name__ == "__main__":
    main()