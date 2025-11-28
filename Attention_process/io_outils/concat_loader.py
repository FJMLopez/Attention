# attention_lib/io/concat_loader.py
import numpy as np
import logging
from typing import List, Dict, Any, Optional

from Attention_process.Classes.Sentence import Sentence
from Attention_process.Classes.Matrix import Matrix
from Attention_process.Classes.AttentionMatrix import AttentionMatrix

logger = logging.getLogger(__name__)

class ConcatModelLoader:
    """
    Chargeur spécifique pour le format de données 'Concaténation'.
    Transforme le JSON brut en objets AttentionMatrix standardisés.
    """

    @staticmethod
    def load_from_json_data(data: Dict[str, Any]) -> List["AttentionMatrix"]:
        """
        Traite un dictionnaire JSON (une entrée du dataset).
        
        Structure attendue de 'data':
        {
            'id': int,
            'src_tokens': str (espace séparés),
            'src_segments_labels': List[int], (pour retrouver les césures contextes)
            'heads_enc_attn': List[List[List[float]]] (Layers x Heads x N x N)
            ...
        }

        Returns:
            List[AttentionMatrix]: Une liste de matrices d'attention (Layer/Head/Phrase vs Contexte).
        """
        # 1. Reconstruction des Phrases (Source)
        # On découpe la "grande phrase" source en plusieurs phrases (Courante + Contextes)
        raw_tokens = data['src_tokens'].split()
        seg_labels = data['src_segments_labels']
        
        # Insertion des <eos> et découpage en objets Sentence
        sentences = ConcatModelLoader._split_into_sentences(raw_tokens, seg_labels, base_id=int(data['id']))
        logger.info(f"ID {data['id']}: Découpé en {len(sentences)} phrases.")
        for i, sentence in enumerate(sentences):
            print(f"sentence {i}: {sentence.tokens}")
        if len(sentences) < 2:
            logger.warning(f"ID {data['id']}: Pas assez de phrases (contexte + courante) trouvées.")
            return []

        # La dernière phrase est la phrase courante (Query), les autres sont le contexte
        current_sentence = sentences[-1]
        context_sentences = sentences[:-1] # Liste des phrases de contexte

        # 2. Chargement de la Matrice Géante (Encodage)
        # Dimensions: [Layers, Heads, batch, Total_Tokens, Total_Tokens]
        enc_attn = np.array(data['heads_enc_attn'], dtype=np.float32)
        print(enc_attn.shape)

        n_layers, n_heads, batch_size, n_rows, n_cols = enc_attn.shape
        
        # Vérification taille totale
        total_len = sum(len(s.tokens) for s in sentences)
        if n_rows != total_len:
            logger.error(f"Mismatch taille matrice ({n_rows}) vs tokens ({total_len}) pour ID {data['id']}")
            # Fallback simple : si mismatch léger dû à <eos> manquant à la fin absolue
            if n_rows == total_len - 1 and sentences[-1].tokens[-1] == "<eos>":
                 # On retire le dernier eos artificiel pour coller à la matrice
                 logger.warning(f"ID {data['id']}: Retrait du dernier <eos> pour correspondance matrice.")
                 sentences[-1].tokens.pop()
            else:
                 return []

        # 3. Extraction des sous-matrices (AttentionMatrix)
        # On veut souvent visualiser l'attention de la phrase courante (Query) 
        # vers l'ensemble du contexte (Key = Concaténation des phrases de contexte).
        
        # Calcul des indices de la phrase courante dans la grande matrice
        # Les phrases sont dans l'ordre [Ctx_k, ..., Ctx_1, Courante]
        # Donc la courante est à la fin.
        start_idx_crt = sum(len(s.tokens) for s in context_sentences)
        end_idx_crt = start_idx_crt + len(current_sentence.tokens)
        
        # Pour le contexte (Keys), on prend tout ce qui précède la phrase courante
        # OU on peut vouloir traiter chaque phrase de contexte séparément.
        # Ici, je propose de concaténer tout le contexte en une seule "Sentence" virtuelle pour la visualisation standard.
        
        full_context_tokens = []
        for s in context_sentences:
            full_context_tokens.extend(s.tokens)
            
        # Création d'une sentence "Contexte Global"
        # ID arbitraire, ou basé sur le premier contexte
        global_context_sentence = Sentence(
            tokens=full_context_tokens,
            system_id=context_sentences[0].system_id if context_sentences else -1
        )
        
        output_matrices = []

        for l in range(n_layers):
            for h in range(n_heads):
                # Extraction du bloc [Lignes Courante, Colonnes Contexte]
                # Slice numpy: [layer, head, range_lignes, range_cols]
                sub_matrix_data = enc_attn[l, h, 0, start_idx_crt:end_idx_crt, 0:start_idx_crt]                
                # Si le contexte est vide (cas rare), on ignore
                if sub_matrix_data.shape[1] == 0:
                    continue

                mat_obj = Matrix(sub_matrix_data)
                
                att_mat = AttentionMatrix(
                    layer_id=l,
                    head_id=h,
                    matrix=mat_obj,
                    row_sentence=current_sentence, # Query
                    col_sentence=global_context_sentence, # Key (Contexte entier)
                    source_model="ConcatModel"
                )
                output_matrices.append(att_mat)
                
        return output_matrices

    @staticmethod
    def _split_into_sentences(raw_tokens: List[str], seg_labels: List[int], base_id: int) -> List[Sentence]:
        """
        Découpe les tokens en phrases.
        CORRECTIF : Si seg_labels[i] != seg_labels[i+1], cela signifie que seg_labels[i] 
        est le label du <eos> (absent du raw_tokens).
        Par conséquent, raw_tokens[i] est le PREMIER token de la phrase SUIVANTE.
        """
        sentences: List[Sentence] = []
        current_tokens: List[str] = []
        is_boundary:bool = False
        
        if not raw_tokens or not seg_labels:
            return []

        # limit = min(len(raw_tokens), len(seg_labels))
        limit: int = len(seg_labels)
        shift: int = 0

        for i in range(limit):
            if is_boundary:
                is_boundary = False
            else :
                tok = raw_tokens[i - shift]
                label = seg_labels[i]
                
                # On regarde si le label change au token SUIVANT
                is_boundary = False
                if i < limit - 1:
                    if seg_labels[i+1] != label:
                        is_boundary = True
                
                if is_boundary:
                    # CAS CRITIQUE : "It" (idx 2) a le label 3, mais le suivant a 2.
                    # Le label 3 appartient à l'EOS implicite. "It" appartient à la phrase 2.
                    
                    # 1. On termine la phrase PRÉCÉDENTE (sans "It")
                    shift += 1
                    if len(current_tokens) > 0: # Si la phrase n'est pas vide
                        # Alors on la sauvegarde
                        current_tokens.append("<eos>")
                        sentences.append(Sentence(tokens=current_tokens, system_id=base_id - label))
                    
                    # 2. On commence la NOUVELLE phrase avec "It"
                    current_tokens = [tok]
                    
                else:
                    # Cas normal : on accumule dans la phrase courante
                    current_tokens.append(tok)
                
        # Sauvegarde de la dernière phrase restée en mémoire
        if current_tokens:
            # On utilise le dernier label vu pour l'ID
            current_tokens.append("<eos>")
            sentences.append(Sentence(tokens=current_tokens, system_id=base_id - seg_labels[-1]))
        logger.debug(f"Découpage en {len(sentences)} phrases effectué.")
        for sentence in sentences:
            logger.debug(f"Phrase ID {sentence.system_id} (len: {len(sentence.tokens)}): {' '.join(sentence.tokens)}")
        return sentences


if __name__ == "__main__":
    import doctest; doctest.testmod()




