import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional

from Attention_process.Classes.Sentence import Sentence
from Attention_process.Classes.Matrix import Matrix
from Attention_process.Classes.AttentionMatrix import AttentionMatrix

logger = logging.getLogger(__name__)

class MultiEncoderLoader:
    """
    Chargeur modulaire pour les modèles Multi-Encoder (Hierarchical Attention).
    Sépare le parsing des phrases, des matrices Word-Level et Sentence-Level.
    """

    @staticmethod
    def load_from_json_data(data: Dict[str, Any], padding_mark: str = "<pad>", eos_token: str = "<eos>") -> Dict[str, List[AttentionMatrix]]:
        """
        Point d'entrée principal. Orchestre le chargement.
        """
        sys_id = int(data.get("id", 0))

        # 1. Parsing des Phrases (Crt + Contextes)
        crt_snt, valid_ctx_snts, valid_ctx_indices = MultiEncoderLoader._parse_sentences(
            data, sys_id, padding_mark, eos_token
        )

        if not crt_snt:
            logger.warning(f"ID {sys_id}: Phrase courante invalide.")
            return {'word_level': [], 'sentence_level': []}
        
        if not valid_ctx_snts:
            logger.debug(f"ID {sys_id}: Aucun contexte valide.")
            return {'word_level': [], 'sentence_level': []}

        # 2. Traitement Word-Level (Heads)
        wl_matrices = MultiEncoderLoader._process_word_level(
            data.get("heads", []), crt_snt, valid_ctx_snts, valid_ctx_indices, sys_id
        )

        # 3. Traitement Sentence-Level (SL_matrice)
        sl_matrices = MultiEncoderLoader._process_sentence_level(
            data.get("SL_matrice", []), crt_snt, len(data.get("ctxs", [])), valid_ctx_indices, sys_id
        )

        return {
            'word_level': wl_matrices,
            'sentence_level': sl_matrices
        }

    # -------------------------------------------------------------------------
    # Méthodes Privées de Traitement
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_sentences(data: Dict[str, Any], sys_id: int, padding_mark: str, eos_token: str) -> Tuple[Optional[Sentence], List[Sentence], List[int]]:
        """
        Extrait et valide les objets Sentence pour la phrase courante et les contextes.
        Gère l'ajout automatique de <eos> si manquant.

        """
        # A. Phrase Courante
        crt_tokens = data.get("crt", [])
        if MultiEncoderLoader._is_effectively_empty(crt_tokens, padding_mark):
            return None, [], []
        
        # Ajout eos si manquant
        if crt_tokens and crt_tokens[-1] != eos_token:
            crt_tokens.append(eos_token)
            
        crt_snt = Sentence(tokens=crt_tokens, system_id=sys_id)

        # B. Contextes
        raw_ctxs = data.get("ctxs", [])
        valid_ctx_indices = []
        valid_ctx_snts = []

        for k, tokens in enumerate(raw_ctxs):
            if not MultiEncoderLoader._is_effectively_empty(tokens, padding_mark):
                # Ajout eos si manquant
                if tokens and tokens[-1] != eos_token:
                    tokens.append(eos_token)
                
                # ID arbitraire : id - 1 - k
                ctx_obj = Sentence(tokens=tokens, system_id=sys_id - 1 - k)
                valid_ctx_snts.append(ctx_obj)
                valid_ctx_indices.append(k)

        return crt_snt, valid_ctx_snts, valid_ctx_indices

    @staticmethod
    def _process_word_level(raw_heads: List[Any], 
                            crt_snt: Sentence, 
                            valid_ctx_snts: List[Sentence], 
                            valid_ctx_indices: List[int], 
                            sys_id: int) -> List[AttentionMatrix]:
        """
        Traite les matrices d'attention mot-à-mot (Word Level).
        Gère le redimensionnement et le slicing en cas de padding excessif.
        """
        matrices = []

        for i, original_k in enumerate(valid_ctx_indices):
            if original_k >= len(raw_heads):
                continue
            
            ctx_matrices_data = raw_heads[original_k]
            ctx_obj = valid_ctx_snts[i]

            for head_id, mat_data in enumerate(ctx_matrices_data):
                mat_array = np.array(mat_data, dtype=np.float32)
                
                # 1. Nettoyage des dimensions (Squeeze intelligent)
                target_shape = (len(crt_snt), len(ctx_obj))
                
                if mat_array.ndim > 2:
                    mat_array = np.squeeze(mat_array)
                
                # 2. Ajustement des colonnes (Padding contexte)
                # Si la matrice est plus large que la phrase de contexte, on coupe à droite
                # (Car les tokens de padding sont souvent à la fin ou ignorés dans Sentence)
                current_cols = mat_array.shape[1] if mat_array.ndim == 2 else 0
                expected_cols = len(ctx_obj)

                if current_cols > expected_cols:
                    # On garde les N derniers tokens si padding à gauche, ou N premiers si padding à droite.
                    # Hypothèse standard : on aligne sur la fin (souvent le cas avec <eos>)
                    # MAIS attention : si la matrice inclut du padding à la fin, il faut couper à la fin.
                    # Ici, on coupe pour garder les 'expected_cols' colonnes.
                    # Votre code original faisait : mat_array[:, -len(ctx_obj.tokens):] (garde la fin)
                    mat_array = mat_array[:, -expected_cols:]
                
                # 3. Ajustement des lignes (Padding courante)
                current_rows = mat_array.shape[0] if mat_array.ndim == 2 else 0
                expected_rows = len(crt_snt)

                if current_rows > expected_rows:
                    mat_array = mat_array[-expected_rows:, :]

                # 4. Validation Finale
                if mat_array.shape != target_shape:
                    logger.debug(f"ID {sys_id} Ctx {original_k} H{head_id}: "
                                 f"Mismatch final {mat_array.shape} vs {target_shape}. Skip.")
                    continue

                att_mat = AttentionMatrix(
                    layer_id=0,
                    head_id=head_id,
                    matrix=Matrix(mat_array),
                    row_sentence=crt_snt,
                    col_sentence=ctx_obj,
                    source_model=f"MultiEnc_WL_Ctx{original_k}"
                )
                matrices.append(att_mat)
        
        return matrices

    @staticmethod
    def _process_sentence_level(raw_sl_data: Any, 
                                crt_snt: Sentence, 
                                total_ctx_count: int, 
                                valid_ctx_indices: List[int], 
                                sys_id: int) -> List[AttentionMatrix]:
        """
        Traite les matrices d'attention hiérarchique (Sentence Level).
        """
        raw_sl_data = np.array(raw_sl_data, dtype=np.float32)
        matrices = []

        # Squeeze dimension inutile (ex: [Heads, Rows, 1, Cols])
        if raw_sl_data.ndim == 4:
            raw_sl_data = np.squeeze(raw_sl_data, axis=2)

        if raw_sl_data.ndim != 3:
            logger.error(f"ID {sys_id}: Format SL_matrice invalide {raw_sl_data.shape}")
            return []

        n_heads, n_rows, n_cols_total = raw_sl_data.shape

        # Validation Lignes
        if n_rows != len(crt_snt):
            # Tentative de slice si la matrice est plus grande (padding)
            if n_rows > len(crt_snt):
                raw_sl_data = raw_sl_data[:, -len(crt_snt):, :]
            else:
                logger.error(f"ID {sys_id}: SL mismatch rows {n_rows} vs {len(crt_snt)}")
                return []

        # Validation Colonnes
        if n_cols_total != total_ctx_count:
             logger.error(f"ID {sys_id}: SL mismatch cols {n_cols_total} vs {total_ctx_count}")
             # Pas de return ici, on essaie quand même de filtrer si possible

        # Filtrage des colonnes (garder uniquement les contextes valides)
        if len(valid_ctx_indices) <= n_cols_total:
            # On utilise le slicing numpy avancé avec la liste d'indices
            filtered_data = raw_sl_data[:, :, valid_ctx_indices]
        else:
            filtered_data = raw_sl_data

        # Création Phrase Artificielle pour les colonnes
        ctx_tokens = [f"Context_{k}" for k in valid_ctx_indices]
        sl_col_snt = Sentence(tokens=ctx_tokens, system_id=-999)

        for h_id in range(n_heads):
            mat_data = filtered_data[h_id]
            
            att_mat = AttentionMatrix(
                layer_id=0,
                head_id=h_id,
                matrix=Matrix(mat_data),
                row_sentence=crt_snt,
                col_sentence=sl_col_snt,
                source_model="MultiEnc_SL"
            )
            matrices.append(att_mat)

        return matrices

    @staticmethod
    def _is_effectively_empty(tokens: List[str], padding_mark: str) -> bool:
        if not tokens:
            return True
        non_pad = [t for t in tokens if t != padding_mark]
        return len(non_pad) == 0