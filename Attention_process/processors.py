# processors.py
import numpy as np
import logging
from enum import Enum
from typing import Callable
from models import AttentionMatrix

logger = logging.getLogger("AttentionProcessor")

class AggregationMethod(Enum):
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"

class BPEMerger:
    """Gère la fusion des BPE sur des matrices rectangulaires."""

    @staticmethod
    def _is_continuation(token: str) -> bool:
        # Adaptez le marqueur selon votre tokenizer (ex: "##", "Ġ")
        return token.startswith("##")

    @staticmethod
    def _get_merge_groups(tokens: list[str]) -> tuple[list[str], list[list[int]]]:
        """
        Analyse une liste de tokens et retourne :
        1. La liste des nouveaux mots fusionnés.
        2. La liste des groupes d'indices correspondants.
        """
        groups: list[list[int]] = []
        current_group: list[int] = []

        for i, token in enumerate(tokens):
            if BPEMerger._is_continuation(token) and current_group:
                current_group.append(i)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [i]
        if current_group:
            groups.append(current_group)

        new_tokens_str = []
        for grp in groups:
            # Reconstruction du mot propre
            words = [tokens[idx].replace("##", "") for idx in grp]
            new_tokens_str.append("".join(words))
            
        return new_tokens_str, groups

    @staticmethod
    def merge(att_data: AttentionMatrix, method: AggregationMethod = AggregationMethod.MEAN) -> AttentionMatrix:
        logger.debug(f"Début fusion BPE (Rectangulaire) pour {att_data}")
        
        # 1. Calcul des groupes pour les Lignes (Phrase Courante)
        new_row_tokens, row_groups = BPEMerger._get_merge_groups(att_data.row_tokens)
        
        # 2. Calcul des groupes pour les Colonnes (Contexte)
        new_col_tokens, col_groups = BPEMerger._get_merge_groups(att_data.col_tokens)

        # 3. Réduction de la matrice
        original_matrix = att_data.matrix
        new_matrix = np.zeros((len(row_groups), len(col_groups)), dtype=original_matrix.dtype)

        # Itération sur les blocs rectangulaires
        for i, r_grp in enumerate(row_groups):
            for j, c_grp in enumerate(col_groups):
                # Extraction du sous-bloc (intersection des BPE lignes et BPE colonnes)
                sub_block = original_matrix[np.ix_(r_grp, c_grp)]
                
                if method == AggregationMethod.SUM:
                    val = np.sum(sub_block)
                elif method == AggregationMethod.MEAN:
                    val = np.mean(sub_block)
                elif method == AggregationMethod.MAX:
                    val = np.max(sub_block)
                else:
                    val = np.mean(sub_block)
                
                new_matrix[i, j] = val

        logger.info(f"Fusion terminée: {att_data.shape} -> {new_matrix.shape}")
        
        return AttentionMatrix(
            layer_id=att_data.layer_id,
            head_id=att_data.head_id,
            row_tokens=new_row_tokens,
            col_tokens=new_col_tokens,
            matrix=new_matrix,
            source_model=att_data.source_model
        )

class Normalizer:
    @staticmethod
    def min_max_normalize(matrix: np.ndarray) -> np.ndarray:
        min_val, max_val = np.min(matrix), np.max(matrix)
        if max_val - min_val == 0: return np.zeros_like(matrix)
        return (matrix - min_val) / (max_val - min_val)

    @staticmethod
    def row_stochastic(matrix: np.ndarray) -> np.ndarray:
        """
        Normalise pour que la somme des poids d'attention de la phrase courante 
        vers le contexte soit égale à 1.
        """
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0 
        return matrix / row_sums

    @staticmethod
    def apply(att_data: AttentionMatrix, func: Callable[[np.ndarray], np.ndarray]) -> AttentionMatrix:
        logger.debug(f"Application normalisation {func.__name__}")
        # Copie explicite pour immutabilité
        new_obj = AttentionMatrix(
            layer_id=att_data.layer_id,
            head_id=att_data.head_id,
            row_tokens=att_data.row_tokens[:],
            col_tokens=att_data.col_tokens[:],
            matrix=func(att_data.matrix),
            source_model=att_data.source_model
        )
        return new_obj