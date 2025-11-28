# attention_lib/io/concat_loader.py
import numpy as np
import logging
from typing import List, Dict, Any, Optional

from Attention_process.Classes.Sentence import Sentence
from Attention_process.Classes.Matrix import Matrix
from Attention_process.Classes.ConcatAttentionMatrix import ConcatAttentionMatrix

logger = logging.getLogger(__name__)

class ConcatModelLoader:
    """
    Chargeur spécifique pour le format de données 'Concaténation'.
    """

    @staticmethod
    def load_from_json_data(data: Dict[str, Any]) -> List[ConcatAttentionMatrix]:
        # 1. Parsing des phrases (Identique au code précédent corrigé)
        raw_tokens = data['src_tokens'].split()
        seg_labels = data['src_segments_labels']
        sentences = ConcatModelLoader._split_into_sentences(raw_tokens, seg_labels, base_id=int(data['id']))
        
        if len(sentences) < 2:
            logger.warning(f"ID {data['id']}: Insuffisant (Contextes + Courante requis).")
            return []

        # Séparation : La dernière est la courante, le reste est le contexte
        current_sentence = sentences[-1]
        context_sentences = sentences[:-1]

        # 2. Matrice
        enc_attn = np.array(data['heads_enc_attn'], dtype=np.float32)
        
        # Gestion dimensions 4D ou 5D selon votre format (Batch ou pas)
        if enc_attn.ndim == 5:
             # [Layers, Heads, Batch, Rows, Cols] -> On prend Batch 0
             n_layers, n_heads, _, n_rows, n_cols = enc_attn.shape
             enc_attn = enc_attn[:, :, 0, :, :]
        else:
             n_layers, n_heads, n_rows, n_cols = enc_attn.shape

        # Validation taille
        total_len = sum(len(s.tokens) for s in sentences)
        if n_rows != total_len:
             # Tentative de correction simple pour <eos> final manquant
             if n_rows == total_len - 1 and sentences[-1].tokens[-1] == "<eos>":
                 logger.warning(f"ID {data['id']}: Correction automatique - Suppression <eos> final manquant.")
                 sentences[-1].tokens.pop()
                 # Recalculs après correction
                 current_sentence = sentences[-1]
                 total_len -= 1
             
             if n_rows != total_len:
                logger.error(f"ID {data['id']}: Mismatch dimensions Matrice({n_rows}) vs Tokens({total_len}). Skip.")
                return []

        # 3. Extraction
        # On extrait la zone : Lignes = Phrase Courante, Colonnes = Tout le contexte
        # Les contextes sont au début (0 -> K), la phrase courante à la fin.
        
        len_ctx_total = sum(len(s.tokens) for s in context_sentences)
        len_current = len(current_sentence.tokens)
        
        # Indices pour slicer la grande matrice
        start_row = len_ctx_total
        end_row = len_ctx_total + len_current
        start_col = 0
        end_col = len_ctx_total

        output_matrices = []

        for l in range(n_layers):
            for h in range(n_heads):
                # Slicing Numpy : [Lignes(Courante), Colonnes(Contextes)]
                sub_data = enc_attn[l, h, start_row:end_row, start_col:end_col]
                
                if sub_data.shape[1] == 0:
                    continue

                mat_obj = Matrix(sub_data)
                # Instanciation de la nouvelle classe spécialisée
                concat_mat = ConcatAttentionMatrix(
                    layer_id=l,
                    head_id=h,
                    matrix=mat_obj,  # Transpose pour aligner lignes/colonnes
                    row_sentence=current_sentence,
                    context_sentences=context_sentences, # Liste complète des contextes
                    source_model="ConcatModel"
                )
                output_matrices.append(concat_mat)
                
        return output_matrices

    @staticmethod
    def _split_into_sentences(raw_tokens: List[str], seg_labels: List[int], base_id: int) -> List[Sentence]:
        """
        Scinde les tokens bruts en phrases basées sur les labels de segments.

        Args:
            raw_tokens (List[str]): Liste des tokens bruts.
            seg_labels (List[int]): Labels de segments correspondants.
            base_id (int): ID de base pour les phrases.

        Returns:
            List[Sentence]: Liste des objets Sentence créés.
        """
        sentences = []
        current_tokens = []
        
        if not raw_tokens or not seg_labels:
            return []
        is_boundary: bool = False
        limit = min(len(raw_tokens), len(seg_labels))
        shift: int = 0
        for i in range(limit):
            if is_boundary :
                is_boundary = False
            else:
                tok = raw_tokens[i-shift]
                label = seg_labels[i]
                
                is_boundary = False
                if i < limit - 1:
                    if seg_labels[i+1] != label:
                        is_boundary = True
                        shift += 1
                
                if is_boundary:
                    current_tokens.append("<eos>")
                    sentences.append(Sentence(tokens=current_tokens, system_id=base_id - label))
                    current_tokens = [tok]
                else:
                    current_tokens.append(tok)
        for j in range(shift):
            current_tokens.append(raw_tokens[limit - shift + j])
        if current_tokens:
            current_tokens.append("<eos>")
            sentences.append(Sentence(tokens=current_tokens, system_id=base_id - seg_labels[-1]))
        return sentences

if __name__ == "__main__":
    import doctest; doctest.testmod()




