import numpy as np
import logging
from typing import List, Dict, Any, Tuple

from Attention_process.Classes.Sentence import Sentence
from Attention_process.Classes.Matrix import Matrix
from Attention_process.Classes.AttentionMatrix import AttentionMatrix

logger = logging.getLogger(__name__)

class MultiEncoderLoader:
    """
    Chargeur spécifique pour les modèles Multi-Encoder (Hierarchical Attention).
    Traite les matrices Word-Level (WL) et Sentence-Level (SL).
    """

    @staticmethod
    def _is_effectively_empty(tokens: List[str], padding_mark: str = "<pad>") -> bool:
        """Vérifie si une phrase est vide ou ne contient que du padding."""
        if not tokens:
            return True
        # On vérifie s'il reste des tokens une fois le padding retiré
        non_pad = [t for t in tokens if t != padding_mark]
        return len(non_pad) == 0

    @staticmethod
    def load_from_json_data(data: Dict[str, Any], padding_mark: str = "<pad>", eos_token: str = "<eos>") -> Dict[str, List[AttentionMatrix]]:
        """
        Charge les données d'un JSON multi-encoder.

        Args:
            data (dict): Le dictionnaire JSON brut.
            padding_mark (str): Le token de padding à ignorer.

        Returns:
            Dict[str, List[AttentionMatrix]]: Un dictionnaire contenant deux listes :
                - 'word_level' : Les matrices d'attention mot-à-mot (Crt -> Ctx_i).
                - 'sentence_level' : Les matrices d'attention hiérarchique (Crt -> [Ctx_0, Ctx_1...]).
        """
        
        # 1. Parsing de la Phrase Courante (CRT)
        crt_tokens = data.get("crt", [])
        sys_id = int(data.get("id", 0))

        if MultiEncoderLoader._is_effectively_empty(crt_tokens, padding_mark):
            logger.warning(f"ID {sys_id}: La phrase courante est vide ou ne contient que du padding. Ignoré.")
            return {'word_level': [], 'sentence_level': []}

        crt_snt = Sentence(tokens=crt_tokens, system_id=sys_id)

        # 2. Parsing des Phrases de Contexte (CTXS)
        # On garde une trace des indices valides (non vides) pour filtrer les matrices
        raw_ctxs = data.get("ctxs", [])
        valid_ctx_indices = [] # Indices originaux (k)
        valid_ctx_snts = []    # Objets Sentence

        for k, tokens in enumerate(raw_ctxs):
            # Si la phrase de contexte est vide/padding pur, on l'ignore
            if not MultiEncoderLoader._is_effectively_empty(tokens, padding_mark):
                # ID système arbitraire pour le contexte (ex: id - 1 - k)
                ctx_obj = Sentence(tokens=tokens, system_id=sys_id - 1 - k)
                valid_ctx_snts.append(ctx_obj)
                valid_ctx_indices.append(k)
        
        if not valid_ctx_snts:
            logger.debug(f"ID {sys_id}: Aucun contexte valide trouvé.")
            return {'word_level': [], 'sentence_level': []}



        # --- A. Traitement Word-Level (Heads) ---
        # Structure JSON : heads[k][h] -> Matrice (N_crt x N_ctx_k)
        # k = indice du contexte, h = indice de la tête
        
        raw_heads = data.get("heads", [])
        wordlevel_matrices = []

        # On itère uniquement sur les contextes valides
        for i, original_k in enumerate(valid_ctx_indices):
            if original_k >= len(raw_heads):
                logger.error(f"ID {sys_id}: Manque de données matrices 'heads' pour le contexte {original_k}")
                continue
            
            ctx_matrices_data = raw_heads[original_k] # Liste des têtes pour ce contexte
            ctx_obj = valid_ctx_snts[i]

            for head_id, mat_data in enumerate(ctx_matrices_data):
                # Conversion Numpy
                mat_array = np.array(mat_data, dtype=np.float32)
                
                # Vérification dimensions
                if mat_array.shape != (len(crt_snt.tokens), len(ctx_obj.tokens)):
                    # Tentative de correction dimension batch (si 3D ou 4D)
                    mat_array = np.squeeze(mat_array)
                
                if mat_array.ndim != 2:
                    logger.warning(f"ID {sys_id} Ctx {original_k} Head {head_id}: Dimensions invalides {mat_array.shape}. Skip.")
                    continue
                mat_matrix = Matrix(mat_array)

                # Validation tokens <eos>
                if crt_snt.tokens[-1] != eos_token:
                    crt_snt.append(eos_token)
                if ctx_obj.tokens[-1] != eos_token:
                    ctx_obj.append(eos_token)
                # Validation dimensions après ajustement tokens
                if mat_matrix.cols != len(ctx_obj.tokens):
                    logger.debug(f"ID {sys_id} Ctx {original_k} Head {head_id}: Mismatch dimensions matrice ({mat_matrix.cols}) vs tokens contexte ({len(ctx_obj.tokens)}).")
                    mat_matrix = Matrix(mat_array[:, -len(ctx_obj.tokens):])


                # Création de l'objet AttentionMatrix
                att_mat = AttentionMatrix(
                    layer_id=0, # Le JSON ne précise pas le layer, on met 0 par défaut
                    head_id=head_id,
                    matrix=mat_matrix,
                    row_sentence=crt_snt,
                    col_sentence=ctx_obj,
                    source_model=f"MultiEnc_WL_Ctx{original_k}" # Tag pour identifier le contexte
                )
                wordlevel_matrices.append(att_mat)

        # --- B. Traitement Sentence-Level (SL_matrice) ---
        # Structure JSON : SL_matrice[h] -> Matrice (N_crt x 1 x N_total_ctx) ou proche
        # Dimensions attendues : [nb_heads, len_crt, 1, nb_total_ctx]
        
        raw_sl_data = np.array(data.get("SL_matrice", []), dtype=np.float32)
        sl_matrices = []

        if raw_sl_data.ndim == 4:
            # Squeeze de la dimension inutile '1'
            # [Heads, Rows, 1, Cols] -> [Heads, Rows, Cols]
            raw_sl_data = np.squeeze(raw_sl_data, axis=2)

        if raw_sl_data.ndim == 3:
            n_heads, n_rows, n_cols_total = raw_sl_data.shape
            
            # Validation des dimensions
            if n_rows != len(crt_snt.tokens):
                logger.error(f"ID {sys_id}: SL_matrice mismatch rows ({n_rows} vs {len(crt_snt.tokens)})")
            elif n_cols_total != len(raw_ctxs):
                 logger.error(f"ID {sys_id}: SL_matrice mismatch cols ({n_cols_total} vs {len(raw_ctxs)} contexts)")
            else:
                # Filtrage des colonnes : on ne garde que les colonnes correspondant aux contextes valides
                # Si on a supprimé le contexte 1 (vide), il faut supprimer la colonne 1 de la matrice SL
                if len(valid_ctx_indices) != n_cols_total:
                    # Slicing numpy pour ne garder que les indices valides
                    filtered_sl_data = raw_sl_data[:, :, valid_ctx_indices]
                else:
                    filtered_sl_data = raw_sl_data

                # Création d'une "Phrase" synthétique pour représenter les Contextes en colonnes
                # Ex: Tokens = ["Ctx_0", "Ctx_2", "Ctx_3"]
                ctx_summary_tokens = [f"Context_{k}" for k in valid_ctx_indices]
                ctx_summary_snt = Sentence(tokens=ctx_summary_tokens, system_id=-999)

                for head_id in range(n_heads):
                    mat_data = filtered_sl_data[head_id]
                    
                    sl_mat = AttentionMatrix(
                        layer_id=0, # SL est souvent une couche unique
                        head_id=head_id,
                        matrix=Matrix(mat_data),
                        row_sentence=crt_snt,     # Query = Mots de la phrase courante
                        col_sentence=ctx_summary_snt, # Key = Les phrases de contexte elles-mêmes
                        source_model="MultiEnc_SL"
                    )
                    sl_matrices.append(sl_mat)
        else:
            logger.error(f"ID {sys_id}: Format SL_matrice non reconnu {raw_sl_data.shape}")

        return {
            'word_level': wordlevel_matrices,
            'sentence_level': sl_matrices
        }