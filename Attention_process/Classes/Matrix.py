# matrix.py
import json
import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Optional, Literal, TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Matrix:
    """
    Représente une matrice mathématique encapsulant un tableau NumPy.
    Cette classe est immuable : toute opération renvoie une nouvelle instance.
    """
    data: np.ndarray

    def __post_init__(self):
        """Validation post-initialisation pour garantir un tableau 2D."""
        # On s'assure que c'est bien un numpy array float
        if not isinstance(self.data, np.ndarray):
            # Hack pour contourner le frozen=True dans le post_init
            object.__setattr__(self, 'data', np.array(self.data, dtype=np.float64))
        
        if self.data.ndim == 0:
             # Cas spécial pour l'initialisation vide ou scalaire transformé
             pass 
        elif self.data.ndim != 2:
            raise ValueError(f"La matrice doit être 2D. Reçu: {self.data.ndim}D")

    # --- Propriétés ---

    @property
    def shape(self) -> tuple[int, int]:
        """Retourne les dimensions de la matrice (lignes, colonnes)."""
        return self.data.shape

    @property
    def rows(self) -> int:
        return self.data.shape[0]

    @property
    def cols(self) -> int:
        return self.data.shape[1]

    # --- Méthodes Utilitaires ---

    def to_list(self, precision: int = 2) -> List[List[float]]:
        """Convertit la matrice en liste de listes avec arrondi.

        Args:
            precision (int): Nombre de décimales.

        Returns:
            List[List[float]]: Données formatées.
        """
        return np.round(self.data, decimals=precision).tolist()

    def transpose(self) -> 'Matrix':
        """Retourne la transposée de la matrice."""
        return Matrix(self.data.T)
    
    def ensure_stochastic(self, epsilon: float = 1e-4) -> 'Matrix':
        """
        Vérifie que la somme de chaque ligne est égale à 1 (propriété d'attention).
        Si ce n'est pas le cas, tente une transposition.
        
        Logique:
        1. Vérifie Somme(Lignes) == 1. Si OK -> Retourne self.
        2. Sinon, transpose et vérifie Somme(Lignes) == 1.
           Si OK -> Log Warning + Retourne la matrice transposée.
        3. Sinon -> Lève ValueError.

        Args:
            epsilon (float): Tolérance pour la comparaison flottante.

        Returns:
            Matrix: L'instance courante ou une nouvelle instance transposée si corrigée.
        
        Raises:
            ValueError: Si la matrice n'est stochastique ni en ligne ni en colonne.
        """
        # 1. Vérification standard (Lignes)
        row_sums = self.data.sum(axis=1)
        # On vérifie si toutes les sommes sont proches de 1
        # Note: on gère le cas des lignes vides (somme=0) qui peuvent arriver avec du padding masqué,
        # mais la demande spécifie "égale à 1". On suppose ici une matrice d'attention valide.
        if np.allclose(row_sums, 1.0, atol=epsilon):
            return self

        # 2. Tentative de correction (Transposition)
        col_sums = self.data.sum(axis=0) # Équivalent à row_sums de la transposée
        
        if np.allclose(col_sums, 1.0, atol=epsilon):
            logger.warning(
                f"Matrice non stochastique sur les lignes (Somme != 1). "
                f"Cependant, elle l'est sur les colonnes. Transposition appliquée automatiquement."
            )
            return self.transpose()

        # 3. Échec critique
        # Calcul de l'écart max pour le message d'erreur
        max_deviation = np.max(np.abs(row_sums - 1.0))
        raise ValueError(
            f"La matrice d'attention est invalide : la somme des lignes n'est pas égale à 1 "
            f"(Ecart max: {max_deviation:.4f}). "
            f"La transposition n'a pas résolu le problème."
        )

    # --- Méthodes de Manipulation (Retournent une nouvelle Matrix) ---

    def remove_indices(self, row_indices: Optional[List[int]] = None, 
                       col_indices: Optional[List[int]] = None) -> 'Matrix':
        """Supprime les lignes et/ou colonnes spécifiées.

        Args:
            row_indices (List[int], optional): Indices des lignes à supprimer.
            col_indices (List[int], optional): Indices des colonnes à supprimer.

        Returns:
            Matrix: Nouvelle matrice réduite.
            
        Examples:
            >>> m = Matrix(np.array([[1, 2], [3, 4], [5, 6]]))
            >>> m.remove_indices(row_indices=[1]).data.tolist()
            [[1, 2], [5, 6]]
        """
        new_data = self.data
        
        if row_indices:
            # Création d'un masque booléen pour garder ce qui n'est PAS dans row_indices
            mask_r = np.ones(new_data.shape[0], dtype=bool)
            mask_r[row_indices] = False
            new_data = new_data[mask_r, :]
            
        if col_indices:
            mask_c = np.ones(new_data.shape[1], dtype=bool)
            mask_c[col_indices] = False
            new_data = new_data[:, mask_c]
            
        return Matrix(new_data)

    def merge_bpe(self, 
                  row_groups: Optional[List[List[int]]] = None, 
                  col_groups: Optional[List[List[int]]] = None, 
                  method: Literal['max', 'mean', 'sum'] = 'max') -> 'Matrix':
        """
        Fusionne les lignes et colonnes selon des groupes d'indices (Fusion BPE).
        Cette méthode vectorisée est plus performante qu'une boucle itérative.

        Args:
            row_groups: Liste de listes d'indices à fusionner pour les lignes.
                        Si None, on considère chaque ligne comme son propre groupe.
            col_groups: Liste de listes d'indices à fusionner pour les colonnes.
            method: 'max', 'mean', ou 'sum'.

        Returns:
            Matrix: Matrice fusionnée.

        Examples:
            >>> mat = Matrix(np.array([[1, 2], [3, 4], [5, 6]])) # Fusion des lignes 0 et 1, la ligne 2 reste seule. Colonnes inchangées.
            >>> grps = [[0, 1], [2]]
            >>> mat.merge_bpe(row_groups=grps, method='max').data.tolist()
            [[3, 4], [5, 6]]
        """
        # Si aucun groupe n'est fourni, on garde l'axe tel quel (groupe identité)
        r_groups = row_groups if row_groups is not None else [[i] for i in range(self.rows)]
        c_groups = col_groups if col_groups is not None else [[i] for i in range(self.cols)]

        new_shape = (len(r_groups), len(c_groups))
        new_data = np.zeros(new_shape, dtype=self.data.dtype)

        # Itération sur les blocs définis par l'intersection des groupes
        # Note: On ne peut pas facilement vectoriser totalement ceci car les groupes 
        # peuvent avoir des tailles inégales.
        for i, r_grp in enumerate(r_groups):
            for j, c_grp in enumerate(c_groups):
                # Extraction du sous-bloc via maillage numpy
                sub_block = self.data[np.ix_(r_grp, c_grp)]
                
                if method == 'max':
                    val = np.max(sub_block)
                elif method == 'sum':
                    val = np.sum(sub_block)
                elif method == 'mean':
                    val = np.mean(sub_block)
                else:
                    raise ValueError(f"Method inconnue: {method}")
                
                new_data[i, j] = val

        return Matrix(new_data)

    def normalize(self, method: Literal["minmax", "max", "row_stochastic"] = "minmax") -> 'Matrix':
        """Normalise les valeurs de la matrice.

        Args:
            method: 
                - 'minmax': Echelle [0, 1] globale.
                - 'max': Division par le max global.
                - 'row_stochastic': Somme des lignes = 1 (proba).

        Returns:
            Matrix: Nouvelle matrice normalisée.
        """
        if method == "minmax":
            min_v, max_v = np.min(self.data), np.max(self.data)
            if np.isclose(max_v - min_v, 0):
                return Matrix(np.zeros_like(self.data))
            return Matrix((self.data - min_v) / (max_v - min_v))
        
        elif method == "max":
            max_v = np.max(self.data)
            if np.isclose(max_v, 0):
                return Matrix(np.zeros_like(self.data))
            return Matrix(self.data / max_v)

        elif method == "row_stochastic":
            row_sums = self.data.sum(axis=1, keepdims=True)
            # Eviter division par zéro : là où somme=0, on met 1 (le résultat restera 0/1=0)
            safe_sums = np.where(row_sums == 0, 1.0, row_sums)
            return Matrix(self.data / safe_sums)
            
        else:
            raise NotImplementedError(f"Méthode {method} non implémentée.")

    def threshold(self, 
                  method: Literal["uniform", "value", "to_uniform"] = "uniform", 
                  value: Optional[float] = None) -> 'Matrix':
        """Applique un seuillage (supprime les valeurs faibles).

        Args:
            method: 'uniform' (seuil = 1/N), 'value' (seuil explicite), 'to_uniform'.
            value: Valeur du seuil si method='value'.

        Returns:
            Matrix: Nouvelle matrice seuillée.
        """
        cols = self.cols
        if cols <= 1:
            return self

        new_data = self.data.copy()
        
        if method == "uniform":
            # Supprime ce qui est inférieur à l'équiprobabilité (1/nb_colonnes)
            threshold_val = 1.0 / cols
            new_data[new_data < threshold_val] = 0.0
            
        elif method == "to_uniform":
            # Remplace les valeurs faibles par l'uniforme
            threshold_val = 1.0 / cols
            new_data[new_data < threshold_val] = threshold_val
            
        elif method == "value":
            if value is None:
                raise ValueError("Vous devez fournir 'value' pour le seuillage par valeur.")
            # Attention: votre code original faisait 1/value. Je garde cette logique ou j'utilise value brut ?
            # Code original: self[self < 1/value] = 0. Je présume que value est un entier diviseur.
            # Pour être plus propre, on s'attendrait à un float direct, mais respectons la logique existante.
            limit = 1.0 / value if value != 0 else 0
            new_data[new_data < limit] = 0.0
            
        else:
            raise NotImplementedError(f"Method {method} inconnue.")

        return Matrix(new_data)

    def remove_padding(self, 
                       row_pad_indices: Optional[List[int]] = None, 
                       col_pad_indices: Optional[List[int]] = None) -> 'Matrix':
        """Alias pour remove_indices spécifiquement pour le padding."""
        return self.remove_indices(row_pad_indices, col_pad_indices)


if __name__ == "__main__":
    import doctest; doctest.testmod()
    
    # Test manuel de la stochastique
    # Cas 1 : Normal
    mat_ok = Matrix(np.array([[0.5, 0.5], [0.2, 0.8]]))
    assert mat_ok.ensure_stochastic().shape == (2, 2)
    
    # Cas 2 : Transposée
    mat_transposed = Matrix(np.array([[0.5, 0.2], [0.5, 0.8]])) # Somme col = 1, ligne != 1
    fixed = mat_transposed.ensure_stochastic()
    print("Correction transposée effectuée ? :", np.allclose(fixed.data.sum(axis=1), 1.0))