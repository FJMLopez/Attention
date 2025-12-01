import numpy as np
import logging
from typing import List, Literal, Optional

# Imports relatifs (à adapter selon votre structure)
from Attention_process.Classes.ConcatAttentionMatrix import ConcatAttentionMatrix
from Attention_process.Classes.Matrix import Matrix

logger = logging.getLogger(__name__)

class AttentionAggregator:
    """
    Service responsable de l'agrégation de plusieurs matrices d'attention
    en une seule (Moyenne, Max, Somme).
    """

    @staticmethod
    def aggregate(matrices: List[ConcatAttentionMatrix], 
                  method: Literal['mean', 'max', 'sum'] = 'mean',
                  new_layer_id: int = -1,
                  new_head_id: int = -1,
                  custom_model_name: Optional[str] = None) -> ConcatAttentionMatrix:
        """
        Agrège une liste de matrices ConcatAttentionMatrix en une seule.
        
        Conditions :
        - Toutes les matrices doivent avoir la même forme (shape).
        - Toutes les matrices doivent correspondre aux mêmes phrases (row/col sentences).
        
        Args:
            matrices: Liste des objets à fusionner.
            method: 'mean' (moyenne), 'max', 'sum'.
            new_layer_id: ID à assigner à la nouvelle matrice (ex: le layer commun).
            new_head_id: ID à assigner (ex: -1 pour dire 'moyenne').
            custom_model_name: Nom optionnel (ex: "Avg_Last_Layer").

        Returns:
            Une nouvelle instance de ConcatAttentionMatrix.
        """
        if not matrices:
            raise ValueError("La liste des matrices à agréger est vide.")
        
        ref = matrices[0]
        n_mats = len(matrices)
        
        # 1. Validation de la cohérence
        # On suppose que si les tokens sont identiques, la structure est identique.
        for i, mat in enumerate(matrices[1:], 1):
            if mat.shape != ref.shape:
                raise ValueError(f"Mismatch dimension matrice {i}: {mat.shape} vs Ref {ref.shape}")
            # On pourrait vérifier mat.row_sentence == ref.row_sentence, etc.

        logger.info(f"Agrégation de {n_mats} matrices via '{method}'...")

        # 2. Empilement des données (Stacking)
        # On extrait les numpy arrays de chaque objet Matrix wrapper
        # Shape finale du stack : (N_Matrices, Rows, Cols)
        try:
            stack = np.stack([m.matrix.data for m in matrices], axis=0)
        except Exception as e:
            logger.error(f"Erreur lors de l'empilement numpy : {e}")
            raise

        # 3. Calcul Mathématique
        if method == 'mean':
            agg_data = np.mean(stack, axis=0)
        elif method == 'max':
            agg_data = np.max(stack, axis=0)
        elif method == 'sum':
            agg_data = np.sum(stack, axis=0)
        else:
            raise ValueError(f"Méthode non supportée : {method}")

        # 4. Construction du nouvel objet
        # On utilise les phrases de la première matrice comme référence (elles sont identiques)
        
        # Nom du modèle source
        if custom_model_name:
            src_name = custom_model_name
        else:
            src_name = f"{ref.source_model}_Agg_{method}"

        return ConcatAttentionMatrix(
            layer_id=new_layer_id,
            head_id=new_head_id,
            matrix=Matrix(agg_data), # On encapsule le numpy array résultant
            row_sentence=ref.row_sentence, 
            context_sentences=ref.context_sentences,
            source_model=src_name
        )

    @staticmethod
    def aggregate_by_layer(matrices: List[ConcatAttentionMatrix], 
                           target_layer: int, 
                           method: Literal['mean', 'max'] = 'mean') -> Optional[ConcatAttentionMatrix]:
        """Helper pour agréger toutes les têtes d'un layer spécifique."""
        
        # Filtrage
        subset = [m for m in matrices if m.layer_id == target_layer]
        
        if not subset:
            logger.warning(f"Aucune matrice trouvée pour le layer {target_layer}.")
            return None
            
        return AttentionAggregator.aggregate(
            subset, 
            method=method, 
            new_layer_id=target_layer,
            new_head_id=-1, # Convention pour "All Heads"
            custom_model_name=f"Layer{target_layer}_{method.capitalize()}"
        )


if __name__ == "__main__":
    import doctest; doctest.testmod()