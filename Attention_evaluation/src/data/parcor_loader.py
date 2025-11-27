# src/data/parcor_loader.py

import logging
import xml.etree.cElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal

from src.types import CorpusData, Mention, CorpusSentence
from src.utils.tokenization import Tokenizer
from src.utils.alignment import compute_edit_distance

# Initialize module-level logger
logger = logging.getLogger(__name__)

class ParCorLoader:
    """
    A data loader for the ParCorFull 2.0 Corpus.
    
    This class handles the complexity of reading the corpus's specific directory structure,
    parsing proprietary XML formats for words and coreference chains, and aligning
    raw text with tokenized text using edit distance.
    """

    # Hardcoded prefixes from the original script defining the dataset split
    DISCO_PREFIXES = [
        '000_1756', '001_1819', '002_1825', '003_1894', '005_1938', 
        '006_1950', '007_1953', '009_2043', '010_205', '011_2053'
    ]
    
    NEWS_PREFIXES = [
        '03', '04', '05', '07', '08', '09', '10', '13', '16', 
        '17', '18', '19', '20', '21', '22', '23', '24', '25'
    ]

    def __init__(self, base_path: Path):
        """
        Initialize the loader.

        Args:
            base_path (Path): The root directory of the ParCorFull corpus 
                              (e.g., /.../parcor-full/corpus/).
        """
        self.base_path = base_path
        if not self.base_path.exists():
            logger.error(f"ParCorFull base path does not exist: {self.base_path}")

    def _pref2raw(self, prefix: str) -> str:
        """
        Converts a file ID prefix (e.g., '000_1756') to a raw file ID (e.g., '01756').
        Used specifically for DiscoMT file naming conventions.
        """
        parts = prefix.split('_')
        if len(parts) != 2:
            return prefix # Fallback
        
        raw_id = '00' + parts[1]
        if len(raw_id) == 5:
            raw_id = '0' + raw_id
        return raw_id

    def _get_span_indices(self, span_str: str) -> List[int]:
        """
        Parses a span string into a list of integer word indices.
        
        Examples:
            "word_1" -> [1]
            "word_1..word_3" -> [1, 2, 3]
            "word_1, word_5" -> [1, 5] (Handles comma separation if present)
            
        Args:
            span_str (str): The value of the 'span' attribute in XML.

        Returns:
            List[int]: A list of integer indices covered by the span.
        """
        # Normalize delimiters: replace commas and double dots with spaces
        # Note: Original script logic implies '..' denotes a range, 
        # but splits on it to find start/end.
        cleaned_str = span_str.replace(',', ' ').replace('..', ' ')
        tokens = cleaned_str.split()
        
        extracted_nums = []
        for token in tokens:
            if 'word_' in token:
                try:
                    # Extract number after 'word_'
                    num = int(token.replace('word_', ''))
                    extracted_nums.append(num)
                except ValueError:
                    logger.warning(f"Failed to parse word index from token: {token}")

        if not extracted_nums:
            return []

        # Logic derived from legacy: if 2 numbers found implying a range
        if len(extracted_nums) > 1 and '..' in span_str:
             # Assume range if '..' was in original string and we have start/end
             start = extracted_nums[0]
             end = extracted_nums[-1]
             return list(range(start, end + 1))
        
        # Otherwise, return individual indices
        return extracted_nums

    def _load_words(self, filepath: Path, prefix: str) -> Dict[str, str]:
        """
        Parses a *_words.xml file.

        Args:
            filepath (Path): Path to the XML file.
            prefix (str): The file ID prefix to prepend to keys.

        Returns:
            Dict[str, str]: Map of '{prefix}-word_{id}' -> 'Word Text'.
        """
        words_map = {}
        if not filepath.exists():
            logger.warning(f"Word file not found: {filepath}")
            return words_map

        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            for word_elem in root:
                w_id = word_elem.attrib.get('id')
                text = word_elem.text
                if w_id and text:
                    key = f"{prefix}-{w_id}"
                    words_map[key] = text
        except ET.ParseError as e:
            logger.error(f"XML Parse Error in {filepath}: {e}")
            
        return words_map

    def _load_coref(self, filepath: Path, prefix: str) -> Tuple[List[Mention], Dict[str, str]]:
        """
        Parses a *_coref_level.xml file.

        Args:
            filepath (Path): Path to the XML file.
            prefix (str): The file ID prefix.

        Returns:
            Tuple: 
                - List[Mention]: List of Mention objects.
                - Dict[str, str]: Lookup map of '{prefix}-word_{id}' -> 'set_id'.
        """
        mentions = []
        words_in_coref = {}
        
        if not filepath.exists():
            logger.warning(f"Coref file not found: {filepath}")
            return mentions, words_in_coref

        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            for markable in root:
                m_id = markable.attrib.get('id', '')
                span_str = markable.attrib.get('span', '')
                coref_class = markable.attrib.get('coref_class', '')
                mention_text = markable.attrib.get('mention', '')  # Note: often empty in XML

                if not span_str:
                    continue

                word_indices = self._get_span_indices(span_str)
                
                # Create types.Mention dataclass instance
                mentions.append(Mention(
                    id=m_id,
                    fileid=prefix,
                    span=span_str,
                    coref_class=coref_class,
                    mention_text=mention_text,
                    word_indices=word_indices
                ))

                # Populate lookup for individual words
                for idx in word_indices:
                    key = f"{prefix}-word_{idx}"
                    words_in_coref[key] = coref_class

        except ET.ParseError as e:
            logger.error(f"XML Parse Error in {filepath}: {e}")

        return mentions, words_in_coref

    def _process_raw_text(self, 
                          raw_lines: List[str], 
                          lang: Literal['EN', 'DE']) -> Dict[str, Tuple[str, List[Tuple]]]:
        """
        Aligns raw text lines with their tokenized versions using the ad-hoc tokenizer
        and edit distance.

        Args:
            raw_lines (List[str]): List of raw strings from the corpus files.
            lang (str): 'EN' or 'DE' to select tokenizer rules.

        Returns:
            Dict: Map of raw_text -> (tokenized_text, alignment_list)
        """
        text_data = {}
        total_wer = [0, 0, 0, 0]  # Ins, Del, Sub, Total

        for id_line, line in enumerate(raw_lines):
            raw_stripped = line.strip()
            if not raw_stripped:
                logger.debug(f"Skipping empty line at index {id_line}. Line: '{line}'")
                continue

            # 1. Tokenize based on language
            if lang == 'EN':
                tok_str = Tokenizer.en_tokenize(raw_stripped)
            else:
                tok_str = Tokenizer.de_tokenize(raw_stripped)

            # 2. Compute Alignment (Ground Truth vs Tokenized)
            # This step ensures we know which raw token corresponds to which word ID later
            er_res = compute_edit_distance(tok_str, raw_stripped)
            
            # er_res is (insertions, deletions, substitutions, reference_length, alignment_ops)
            # Update WER stats
            for i in range(4):
                total_wer[i] += er_res[i]
            
            alignment = er_res[4]
            
            # Store map
            text_data[raw_stripped] = (tok_str, alignment)

        # Log WER statistics for sanity check
        if total_wer[3] > 0:
            wer_score = sum(total_wer[0:3]) / total_wer[3] * 100
            logger.info(f"[{lang}] Raw-to-Tokenized WER: {wer_score:.2f}%")
        
        return text_data

    def _process_raw_text_debug(self, raw_lines: List[str], lang: str) -> Dict[str, Tuple[str, List[Tuple]]]:
        text_data = {}
        total_wer = [0, 0, 0, 0]

        debug_counter = 0 # Compteur pour ne pas spammer

        for index_line, line in enumerate(raw_lines):
            raw_stripped = line.strip()
            if not raw_stripped:
                continue

            # 1. Tokenize
            if lang == 'EN':
                tok_str = Tokenizer.en_tokenize(raw_stripped)
            else:
                tok_str = Tokenizer.de_tokenize(raw_stripped)

            # 2. Compute Alignment
            er_res = compute_edit_distance(tok_str, raw_stripped)
            
            # --- AJOUT DEBUG ICI ---
            # Si le WER de la phrase est non-nul (c'est-à-dire que le raw ne correspond pas au tok)
            # Dans le script original, c'est normal d'avoir quelques différences (ponctuation), 
            # mais si c'est "beaucoup plus important", on veut voir les insertions/suppressions.
            
            n_ins, n_del, n_sub, ref_len, _ = er_res
            sent_errors = n_ins + n_del + n_sub
            
            # Seuil arbitraire : si plus de 20% d'erreur ou plus de 5 erreurs absolues
            if sent_errors > 5 and debug_counter < 10:
                logger.warning(f"HIGH WER DETECTED on line {index_line}: {raw_stripped[:50]}...")
                logger.warning(f"Raw: |{raw_stripped}|")
                logger.warning(f"Tok: |{tok_str}|")
                logger.warning(f"Stats: Ins={n_ins}, Del={n_del}, Sub={n_sub}")
                debug_counter += 1
            # -----------------------

            for i in range(4):
                total_wer[i] += er_res[i]
            
            text_data[raw_stripped] = (tok_str, er_res[4])

        if total_wer[3] > 0:
            wer_score = sum(total_wer[0:3]) / total_wer[3] * 100
            logger.info(f"[{lang}] Raw-to-Tokenized WER: {wer_score:.2f}% (Total Words: {total_wer[3]})")
        
        return text_data

    def load_disco_mt(self, lang: Literal['EN', 'DE']) -> CorpusData:
        """
        Loads the DiscoMT subset of ParCorFull.

        Structure:
            Raw Text: DiscoMT/{lang}/Source/sentence/talk{raw_id}.de-en.{lower_lang}
            Words:    DiscoMT/{lang}/Basedata/{prefix}_words.xml
            Coref:    DiscoMT/{lang}/Markables/{prefix}_coref_level.xml
        """
        logger.info(f"Loading DiscoMT data for language: {lang}")
        
        # Paths setup
        subset_path = self.base_path / "DiscoMT" / lang
        raw_txt_dir = subset_path / "Source" / "sentence"
        word_dir = subset_path / "Basedata"
        markable_dir = subset_path / "Markables"

        all_words = {}
        all_mentions = []
        all_words_in_coref = {}
        raw_lines_accumulator = []

        # 1. Iterate over predefined file prefixes
        for prefix in self.DISCO_PREFIXES:
            # 1a. Load Raw Text
            raw_id = self._pref2raw(prefix)
            suffix = 'en' if lang == 'EN' else 'de'
            # Note: filename pattern is specific: talkXXXXX.de-en.en
            filename = f"talk{raw_id}.de-en.{suffix}"
            raw_file = raw_txt_dir / filename
            
            if raw_file.exists():
                with open(raw_file, encoding='utf-8') as f:
                    lines = f.readlines()
                    raw_lines_accumulator.extend(lines)
            else:
                logger.warning(f"Raw text file missing: {raw_file}")

            # 1b. Load Words
            w_file = word_dir / f"{prefix}_words.xml"
            words = self._load_words(w_file, prefix)
            all_words.update(words)

            # 1c. Load Coref
            c_file = markable_dir / f"{prefix}_coref_level.xml"
            mentions, w_in_c = self._load_coref(c_file, prefix)
            all_mentions.extend(mentions)
            all_words_in_coref.update(w_in_c)

        # 2. Process Text (Tokenize & Align)
        text_map = self._process_raw_text(raw_lines_accumulator, lang)

        return CorpusData(
            text_map=text_map,
            words=all_words,
            coref_mentions=all_mentions,
            words_in_coref=all_words_in_coref
        )

    def load_news(self, lang: Literal['EN', 'DE']) -> CorpusData:
        """
        Loads the News subset of ParCorFull.

        Structure:
            Raw Text: news/{lang}/Source/{prefix}.{lower_lang}.xml (Nested XML structure)
            Words:    news/{lang}/Basedata/{prefix}_words.xml
            Coref:    news/{lang}/Markables/{prefix}_coref_level.xml
        """
        logger.info(f"Loading News data for language: {lang}")

        subset_path = self.base_path / "news" / lang
        raw_txt_dir = subset_path / "Source"
        word_dir = subset_path / "Basedata"
        markable_dir = subset_path / "Markables"

        all_words = {}
        all_mentions = []
        all_words_in_coref = {}
        raw_lines_accumulator = []

        for prefix in self.NEWS_PREFIXES:
            # 1a. Load Raw Text (XML Format specific to News)
            suffix = 'en' if lang == 'EN' else 'de'
            filename = f"{prefix}.{suffix}.xml"
            raw_file = raw_txt_dir / filename
            
            if raw_file.exists():
                try:
                    tree = ET.parse(raw_file)
                    root = tree.getroot()
                    # Structure: <root><p><seg>Text here</seg></p></root>
                    # Iterate all paragraphs and segments
                    for paragraph in root:
                        for seg in paragraph:
                            if seg.text:
                                raw_lines_accumulator.append(seg.text.strip())
                except ET.ParseError as e:
                    logger.error(f"Error parsing news raw XML {raw_file}: {e}")
            else:
                logger.warning(f"Raw news file missing: {raw_file}")

            # 1b. Load Words
            w_file = word_dir / f"{prefix}_words.xml"
            words = self._load_words(w_file, prefix)
            all_words.update(words)

            # 1c. Load Coref
            c_file = markable_dir / f"{prefix}_coref_level.xml"
            mentions, w_in_c = self._load_coref(c_file, prefix)
            all_mentions.extend(mentions)
            all_words_in_coref.update(w_in_c)

        # 2. Process Text (Tokenize & Align)
        text_map = self._process_raw_text(raw_lines_accumulator, lang)

        return CorpusData(
            text_map=text_map,
            words=all_words,
            coref_mentions=all_mentions,
            words_in_coref=all_words_in_coref
        )

        