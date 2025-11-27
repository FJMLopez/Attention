import re
import sys
from pathlib import Path
from src.utils.tokenization import Tokenizer

# --- 1. L'ANCIENNE FONCTION (Copiée telle quelle du script original) ---
def legacy_en_tokenize(text):
    punctuation = ['.', ',', ';', ':', '!', '?', '"', '\'', '[', ']']
    res = text.strip()
    tok_str = ' '.join(re.split(r'(\.\.\.|\.|,|;|:|\!|\?|"|\'|\[|\]|„|“|‚|‘|”|«|»|\(|\))', res))

    tok_str = tok_str.replace(' \' s ', ' \'s ')
    tok_str = tok_str.replace(' \' d ', ' \'d ')
    tok_str = tok_str.replace(' \' m ', ' \'m ')
    tok_str = tok_str.replace('n \' t ', ' n\'t ')
    tok_str = tok_str.replace(' \' re ', ' \'re ')
    tok_str = tok_str.replace(' \' ve ', ' \'ve ')
    tok_str = tok_str.replace(' \' ll ', ' \'ll ')
    tok_str = tok_str.replace(' \' 60s ', ' \'60s ')
    tok_str = tok_str.replace(' \' 70s ', ' \'70s ')
    tok_str = tok_str.replace(' \' 80s ', ' \'80s ')
    tok_str = tok_str.replace(' \' 90s ', ' \'90s ')
    tok_str = tok_str.replace(' D . S ', ' D.S ')
    tok_str = tok_str.replace(' U . S . ', ' U.S. ')
    tok_str = tok_str.replace(' D . C . ', ' D.C. ')
    tok_str = tok_str.replace(' o \' clock', ' o\'clock')
    tok_str = tok_str.replace(' Dr . ', ' Dr. ')
    tok_str = tok_str.replace(' mid- \'90s ', ' mid-\'90s ')
    tok_str = tok_str.replace(' I . P ', ' I.P ')
    tok_str = tok_str.replace(' P . C ', ' P.C ')
    tok_str = re.sub(' (\d+) , (\d\d\d)', ' \\1,\\2', tok_str )
    tok_str = re.sub('(\d+) . (\d+)', '\\1.\\2', tok_str)
    tok_str = tok_str.replace(' Amazon . com', ' Amazon.com')
    tok_str = tok_str.replace(' cannot', ' can not')
    tok_str = tok_str.replace(' U . K .', ' U.K.')
    tok_str = tok_str.replace(' Ph . D . ', ' Ph.D. ')
    tok_str = re.sub(' \' (\d\d) ', ' \'\\1 ', tok_str)
    tok_str = re.sub(' \$(\d+) ', ' $ \\1 ', tok_str)
    tok_str = tok_str.replace(' wanna', ' wan na')
    tok_str = tok_str.replace(' E . T . ', ' E.T. ')
    # Ad hoc processing for a specific sentence:
    if 'unbelievable movie about what' in tok_str:
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
    tok_str = tok_str.replace(' L . A . ', ' L.A. ')
    tok_str = tok_str.replace(' C \' mere ', ' C\'mere ')
    tok_str = tok_str.replace(' \' Cause ', ' \'Cause ')
    tok_str = tok_str.replace(' C \' mon ', ' C\'mon ')
    tok_str = tok_str.replace(' gotta ', ' got ta ')
    tok_str = tok_str.replace(' Oh-- ', ' Oh -- ')
    tok_str = tok_str.replace(' \' cause ', ' \'cause ')
    tok_str = tok_str.replace( ' gonna ', ' gon na ')

    if tok_str[-5:] == 'U.K. ':
        tok_str = tok_str[:-5] + 'U.K .'
    if tok_str[-5:] == 'U.S. ':
        tok_str = tok_str[:-5] + 'U.S .'
    tok_str = tok_str.replace('  ', ' ')
    tok_str = tok_str.replace(' \' alliance of misfits', ' \'alliance of misfits')
    tok_str = tok_str.replace(' St . Petersburg ', ' St. Petersburg ' )
    tok_str = re.sub( '(\d+)\%', '\\1 %', tok_str )
    tok_str = tok_str.replace('can be \' photographed', 'can be \'photographed')
    tok_str = tok_str.replace('or \' captured', 'or \'captured')
    tok_str = tok_str.replace('www . drjoetoday . com', 'www.drjoetoday.com')
    tok_str = tok_str.replace('Stefanie R . Ellis', 'Stefanie R. Ellis')
    tok_str = re.sub('\$(\d+).(\d+)', '$ \\1,\\2', tok_str)
    tok_str = tok_str.replace('a . m .', 'a.m.')
    tok_str = tok_str.replace('p . m .', 'p.m.')
    tok_str = re.sub(' #(\d+)', ' # \\1', tok_str)

    return tok_str.strip()

# --- 2. FONCTION DE TEST ---
def compare_tokenizers(file_path):
    print(f"Comparaison sur le fichier : {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    diff_count = 0
    for idx, line in enumerate(lines):
        raw = line.strip()
        if not raw: continue

        # Version Ancienne
        legacy_tok = legacy_en_tokenize(raw)
        
        # Nouvelle Version
        new_tok = Tokenizer.en_tokenize(raw)

        # Comparaison stricte
        if legacy_tok != new_tok:
            diff_count += 1
            print(f"\n--- Différence Ligne {idx+1} ---")
            print(f"RAW   : {raw}")
            print(f"LEGACY: |{legacy_tok}|")
            print(f"NEW   : |{new_tok}|")
            
            # Afficher le premier caractère différent pour aider
            min_len = min(len(legacy_tok), len(new_tok))
            for i in range(min_len):
                if legacy_tok[i] != new_tok[i]:
                    print(f"Diff at char {i}: '{legacy_tok[i]}' vs '{new_tok[i]}'")
                    print(f"Context: ...{legacy_tok[max(0, i-5):i+5]}...")
                    break
            
            if diff_count >= 5:
                print("\nSTOP : Trop de différences, arrêt du debug.")
                break
    
    if diff_count == 0:
        print("\n✅ Aucune différence trouvée ! Le tokenizer est fidèle.")
    else:
        print(f"\n❌ {diff_count} différences trouvées.")

if __name__ == "__main__":
    # Remplacez ceci par le chemin vers votre fichier test.en
    # Vous pouvez aussi passer le fichier en argument
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/getalp/lopezfab/Attention/k3/GOLD/test.en"
    compare_tokenizers(path)