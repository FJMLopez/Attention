# src/utils/tokenization.py

import re
import logging

# Initialize module logger
logger = logging.getLogger(__name__)

class Tokenizer:
    """
    Ad-hoc tokenization and normalization utilities for the ParCorFull project.

    This class does not implement standard NLP tokenization (like NLTK or SpaCy).
    Instead, it implements a specific set of hard-coded rules and replacement hacks 
    required to match the inconsistent raw text formatting found in the ParCorFull 
    corpus XML files with the output of NMT systems.

    Warning:
        The order of replacements is critical. Do not reorder or remove the 
        "sentence-specific hacks" without verifying alignment performance against 
        the ground truth.
    """

    # Regex for initial punctuation splitting
    # Splits on: ... . , ; : ! ? " ' [ ] and various quote styles, plus parentheses.
    _SPLIT_PATTERN = re.compile(r'(\.\.\.|\.|,|;|:|\!|\?|"|\'|\[|\]|„|“|‚|‘|”|«|»|\(|\))')

    @classmethod
    def en_tokenize(cls, text: str) -> str:
        """
        Tokenizes English text according to ParCorFull conventions.

        Process:
        1. Splits punctuation.
        2. Re-attaches specific English contractions (e.g., " ' s " -> " 's ").
        3. Normalizes known abbreviations (e.g., "U . S ." -> "U.S.").
        4. Applies specific fixes for numbers, decades, and specific sentences 
           found in the corpus that are known to cause alignment issues.

        Args:
            text (str): The raw input sentence.

        Returns:
            str: The tokenized and normalized string.
        """
        res = text.strip()
        
        # 1. Split punctuation
        # This adds spaces around punctuation, creating tokens like "U . S ."
        tok_str = ' '.join(re.split(cls._SPLIT_PATTERN, res))

        # 2. Fix Contractions & Possessives
        # The split above separates "it's" into "it ' s". We put them back.
        replacements = [
            (" ' s ", " 's "),
            (" ' d ", " 'd "),
            (" ' m ", " 'm "),
            ("n ' t ", " n't "),
            (" ' re ", " 're "),
            (" ' ve ", " 've "),
            (" ' ll ", " 'll "),
            (" ' 60s ", " '60s "),
            (" ' 70s ", " '70s "),
            (" ' 80s ", " '80s "),
            (" ' 90s ", " '90s "),
            (" mid- '90s ", " mid-'90s "),
            (" o ' clock", " o'clock"),
            (" cannot", " can not"),
            (" wanna", " wan na"),
            (" gotta ", " got ta "),
            (" gonna ", " gon na "),
            (" ' Cause ", " 'Cause "),
            (" ' cause ", " 'cause "),
            (" C ' mere ", " C'mere "),
            (" C ' mon ", " C'mon "),
        ]

        # 3. Fix Abbreviations
        # Re-joins abbreviations split by punctuation logic.
        replacements += [
            (" D . S ", " D.S "),
            (" U . S . ", " U.S. "),
            (" D . C . ", " D.C. "),
            (" Dr . ", " Dr. "),
            (" I . P ", " I.P "),
            (" P . C ", " P.C "),
            (" U . K .", " U.K."),
            (" Ph . D . ", " Ph.D. "),
            (" E . T . ", " E.T. "),
            (" L . A . ", " L.A. "),
            (" St . Petersburg ", " St. Petersburg "),
            (" Amazon . com", " Amazon.com"),
            (" www . drjoetoday . com", " www.drjoetoday.com"),
            (" Stefanie R . Ellis", " Stefanie R. Ellis"),
            (" a . m .", " a.m."),
            (" p . m .", " p.m."),
        ]

        # Apply simple string replacements
        for old, new in replacements:
            tok_str = tok_str.replace(old, new)

        # 4. Regex replacements for Numbers and Formatting
        # Fix numbers like "1,000" which became "1 , 000"
        tok_str = re.sub(r' (\d+) , (\d\d\d)', r' \1,\2', tok_str)
        # Fix decimals like "1.5" which became "1 . 5"
        tok_str = re.sub(r'(\d+) . (\d+)', r'\1.\2', tok_str)
        # Fix currency "$100" -> "$ 100"
        tok_str = re.sub(r' \$(\d+) ', r' $ \1 ', tok_str)
        tok_str = re.sub(r'\$(\d+).(\d+)', r'$ \1,\2', tok_str)
        # Fix years/apostrophes: " ' 99 " -> " '99 "
        tok_str = re.sub(r' \' (\d\d) ', r' \' \1 ', tok_str)
        # Fix percentages: "10%" -> "10 %"
        tok_str = re.sub(r'(\d+)\%', r'\1 %', tok_str)
        # Fix hashtags: " #1" -> " # 1"
        tok_str = re.sub(r' #(\d+)', r' # \1', tok_str)

        # 5. Sentence-Specific Hacks (Legacy Preservation)
        # These rules exist to fix specific alignment errors in the original corpus
        if 'unbelievable movie about what' in tok_str:
            # Special handling for "E.T." in a specific context
            tt = tok_str.split()
            new_str = []
            first = True
            for t in tt:
                if t == 'E.T.' and first:
                    new_str.append(t)
                    first = False
                elif t == 'E.T.':
                    new_str.append('E.T')
                    new_str.append('.')
                else:
                    new_str.append(t)
            tok_str = ' '.join(new_str)
            
        if 'divorce-crippled family' in tok_str:
            tok_str = tok_str.replace('E.T.', 'E.T .')
            
        tok_str = tok_str.replace(' Oh-- ', ' Oh -- ')
        
        # Suffix cleanups
        if tok_str.endswith('U.K. '):
            tok_str = tok_str[:-5] + 'U.K .'
        if tok_str.endswith('U.S. '):
            tok_str = tok_str[:-5] + 'U.S .'

        # Misc specific string fixes
        misc_fixes = [
            (" ' alliance of misfits", " 'alliance of misfits"),
            ("can be ' photographed", "can be 'photographed"),
            ("or ' captured", "or 'captured"),
        ]
        for old, new in misc_fixes:
            tok_str = tok_str.replace(old, new)

        # Final Cleanup
        tok_str = tok_str.replace('  ', ' ')
        return tok_str.strip()

    @classmethod
    def de_tokenize(cls, text: str) -> str:
        """
        Tokenizes German text according to ParCorFull conventions.
        
        This method first applies the English tokenization (which handles standard 
        punctuation), and then applies German-specific corrections, particularly 
        regarding number formatting (commas vs decimals) and specific abbreviations.

        Args:
            text (str): The raw input sentence.

        Returns:
            str: The tokenized and normalized string.
        """
        # Start with standard English rules for punctuation splitting
        tok_str = cls.en_tokenize(text)

        # 1. German Number formatting and specific phrase corrections
        # In German, 1.80m is often written 1,80m, or inconsistently in the corpus.
        replacements = [
            (' Die sind 1.80 Meter ', ' Die sind 1,80 Meter '),
            (' Wirtschaftswachstum um 1.3 Prozent ', ' Wirtschaftswachstum um 1,3 Prozent '),
            (' du hättest 1.7 Millionen Dokumente ', ' du hättest 1,7 Millionen Dokumente '),
            (' Mindestlohn von 7.25 US-Dollar sei ', ' Mindestlohn von 7,25 US-Dollar sei '),
            (' U.S. -Außenministeriums ', ' U.S.-Außenministeriums '),
            (' Amazon . com ', ' Amazon.com '),
            (' Mio . ', ' Mio. '),
            (' etc .', ' etc.'),
            (' Google X .', ' Google X.'),
            (' N . Negroponte', ' N. Negroponte'),
            (' usw .', ' usw.'),
            (' ABC " -Lieder', ' ABC"-Lieder'),
            (' wie z . B . die Druckmaschine', ' wie z.B. die Druckmaschine'),
            (' Und wenn man sich z . B .  ', ' Und wenn man sich z.B. '),
            (' ist . ¾', ' ist.¾'),
            (' ist heute überall . .', ' ist heute überall ..'),
            (' wie z . B . Ibrahim Böhme', ' wie z. B. Ibrahim Böhme'),
            (' St . Petersburger', ' St. Petersburger'),
            (' mit 0,000 Leuten', ' mit 20.000 Leuten'), # Specific typo fix in corpus
            (' zu Putin : " Danke ', ' zu Putin:"Danke '),
            (' 1.35 Mrd . ', ' 1,35 Mrd. '),
            (' www . drjoetoday . com', ' www.drjoetoday.com' ),
            (' Final Five " -Mannschaftskameradin ', ' Final Five"-Mannschaftskameradin '),
            (' Stefanie R . Ellis', ' Stefanie R. Ellis' ),
            (' von 13.75 Zoll Regen', ' von 13,75 Zoll Regen' ),
            (' Aber in St. Petersburg lautete', ' Aber in St . Petersburg lautete'),
        ]

        # Specific single sentence fix
        if tok_str == 'So .':
            tok_str = 'So.'
            
        if tok_str == 'Z . B . in der Telekommunikation können Sie die gleiche Geschichte über Glasfaser erklären .':
            tok_str = 'Z. B. in der Telekommunikation können Sie die gleiche Geschichte über Glasfaser erklären .'
            
        if tok_str == 'Es bedeutet z . B . , dass wir ausarbeiten müssen ,  wie man Zusammenarbeit und Konkurrenz gleichzeitig unterbringt .':
            tok_str = 'Es bedeutet z. B. , dass wir ausarbeiten müssen ,  wie man Zusammenarbeit und Konkurrenz gleichzeitig unterbringt .'

        # Apply list replacements
        for old, new in replacements:
            tok_str = tok_str.replace(old, new)

        # 2. Regex corrections for German
        # Ordinal numbers: "50 . ten" -> "50.ten"
        tok_str = re.sub(r' (\d+) . (\d+)?ten ', r' \1.\2ten ', tok_str)
        # Approx numbers: "ca . 6000" -> "ca. 6000"
        tok_str = re.sub(r' ca . (\d)+', r' ca. \1', tok_str)
        # Pounds sterling: "£ 50" -> "£ 50" (Normalization)
        tok_str = re.sub(r' £(\d+)', r' £ \1', tok_str)
        # Milligrams: "50mg" -> "50 mg"
        tok_str = re.sub(r' (\d+)mg ', r' \1 mg ', tok_str)
        # Hash marks
        tok_str = re.sub(r' #(\d+) ', r' # \1 ', tok_str)

        return tok_str.strip()