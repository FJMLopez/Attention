# attention_lib/models/attention_matrix.py
import copy
from dataclasses import dataclass
from typing import Optional, Literal, List, Union

# Imports relatifs (supposant la structure de dossier définie précédemment)
from Attention_process.Classes.Matrix import Matrix
from Attention_process.Classes.Sentence import Sentence
from Attention_process.services.MatrixExporter import MatrixExporter

@dataclass
class AttentionMatrix:
    """
    Classe orchestre qui lie une Matrice mathématique à deux Phrases (Ligne/Query et Colonne/Context).
    Elle assure la cohérence sémantique lors des transformations (BPE, Padding).
    
    Attributes:
        layer_id (int): Numéro de la couche (Layer).
        head_id (int): Numéro de la tête d'attention (Head).
        matrix (Matrix): L'objet mathématique contenant les poids.
        row_sentence (Sentence): La phrase correspondant aux lignes (Phrase courante / Query).
        col_sentence (Sentence): La phrase correspondant aux colonnes (Contexte / Key).
        source_model (Optional[str]): Nom du modèle source (ex: 'Bert', 'Camembert').
    """
    layer_id: int
    head_id: int
    matrix: Matrix
    row_sentence: Sentence
    col_sentence: Sentence
    source_model: Optional[str] = None

    def __post_init__(self):
        """Validation stricte des dimensions entre la matrice et les phrases."""
        n_rows = len(self.row_sentence.tokens)
        n_cols = len(self.col_sentence.tokens)
        
        if self.matrix.shape != (n_rows, n_cols):
            raise ValueError(
                f"Incohérence dimensions [L{self.layer_id}H{self.head_id}]: "
                f"Matrice {self.matrix.shape} vs Tokens [{n_rows}x{n_cols}].\n"
                f"Row Tokens: {self.row_sentence.tokens}\n"
                f"Col Tokens: {self.col_sentence.tokens}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Retourne la forme de la matrice (Lignes, Colonnes)."""
        return self.matrix.shape

    # --- Méthodes de Transformation (Renvoient une nouvelle instance) ---

    def merge_bpe(self, 
                  method: Literal['max', 'mean', 'sum'] = 'max',
                  bpe_mark: str = '@@') -> 'AttentionMatrix':
        """
        Applique la fusion BPE simultanément sur les phrases et la matrice.
        
        Args:
            method: Méthode d'agrégation des poids ('max', 'mean', 'sum').
            bpe_mark: Le marqueur de fin de token BPE (ex: '@@').

        Returns:
            AttentionMatrix: Une nouvelle instance avec les tokens fusionnés et la matrice réduite.
        """
        # 1. Calcul des groupes d'indices (Logique Sentence)
        # Note : On récupère les groupes complets (singletons + fusions)
        row_groups = Sentence.list_fusion_bpe(self.row_sentence.tokens, BPE_mark=bpe_mark)
        col_groups = Sentence.list_fusion_bpe(self.col_sentence.tokens, BPE_mark=bpe_mark)

        # 2. Réduction Mathématique (Logique Matrix)
        # La matrice est immuable, elle renvoie une nouvelle instance réduite
        new_matrix = self.matrix.merge_bpe(row_groups, col_groups, method=method)

        # 3. Mise à jour Sémantique (Logique Sentence)
        # On copie les phrases car Sentence est mutable et on veut renvoyer un nouvel objet AttentionMatrix
        new_row_snt = copy.deepcopy(self.row_sentence)
        new_col_snt = copy.deepcopy(self.col_sentence)

        # On applique la fusion sur les copies (met à jour les tokens internes)
        new_row_snt.fusion_bpe(list_groupes_bpe=row_groups, BPE_mark=bpe_mark)
        new_col_snt.fusion_bpe(list_groupes_bpe=col_groups, BPE_mark=bpe_mark)

        return AttentionMatrix(
            layer_id=self.layer_id,
            head_id=self.head_id,
            matrix=new_matrix,
            row_sentence=new_row_snt,
            col_sentence=new_col_snt,
            source_model=self.source_model
        )

    def remove_padding(self, padding_mark: str = "<pad>") -> 'AttentionMatrix':
        """
        Supprime les lignes et colonnes correspondant aux tokens de padding.
        
        Args:
            padding_mark: Le token considéré comme padding.
            
        Returns:
            AttentionMatrix: Une nouvelle instance sans padding.
        """
        # 1. Identification des indices à supprimer
        row_pad_indices = Sentence.list_suppr_pad(self.row_sentence.tokens, padding_mark=padding_mark)
        col_pad_indices = Sentence.list_suppr_pad(self.col_sentence.tokens, padding_mark=padding_mark)

        if not row_pad_indices and not col_pad_indices:
            return self # Rien à faire

        # 2. Réduction de la matrice (indices supprimés)
        new_matrix = self.matrix.remove_indices(row_indices=row_pad_indices, col_indices=col_pad_indices)

        # 3. Mise à jour des phrases
        new_row_snt = copy.deepcopy(self.row_sentence)
        new_col_snt = copy.deepcopy(self.col_sentence)

        new_row_snt.suppr_pad(list_index=row_pad_indices)
        new_col_snt.suppr_pad(list_index=col_pad_indices)

        return AttentionMatrix(
            layer_id=self.layer_id,
            head_id=self.head_id,
            matrix=new_matrix,
            row_sentence=new_row_snt,
            col_sentence=new_col_snt,
            source_model=self.source_model
        )

    def normalize(self, method: Literal["minmax", "max", "row_stochastic"] = "minmax") -> 'AttentionMatrix':
        """
        Applique une normalisation sur la matrice.
        Les phrases restent inchangées.
        """
        return AttentionMatrix(
            layer_id=self.layer_id,
            head_id=self.head_id,
            matrix=self.matrix.normalize(method), # Renvoie une nouvelle Matrix
            row_sentence=copy.deepcopy(self.row_sentence),
            col_sentence=copy.deepcopy(self.col_sentence),
            source_model=self.source_model
        )
    
    def threshold(self, 
                  method: Literal["uniform", "value", "to_uniform"] = "uniform", 
                  value: Optional[float] = None) -> 'AttentionMatrix':
        """Applique un seuillage sur la matrice."""
        return AttentionMatrix(
            layer_id=self.layer_id,
            head_id=self.head_id,
            matrix=self.matrix.threshold(method, value),
            row_sentence=copy.deepcopy(self.row_sentence),
            col_sentence=copy.deepcopy(self.col_sentence),
            source_model=self.source_model
        )

    # --- Méthodes d'Entrée / Sortie ---

    def save(self, 
             output_dir: str, 
             format: Literal['json', 'tsv', 'xlsx'] = 'json',
             precision: int = 2,
             filename_suffix: str = "") -> None:
        """
        Sauvegarde l'objet dans le format spécifié via MatrixExporter.
        
        Args:
            output_dir: Dossier de destination.
            format: 'json', 'tsv' ou 'xlsx'.
            precision: Nombre de décimales.
            filename_suffix: Suffixe optionnel pour le nom du fichier.
        """
        # Construction du nom de fichier standardisé
        # Ex: L0_H1_Bert.json
        model_part = f"_{self.source_model}" if self.source_model else ""
        filename = f"L{self.layer_id}_H{self.head_id}{model_part}{filename_suffix}"

        if format == 'json':
            MatrixExporter.to_json(self.matrix, self.row_sentence, self.col_sentence, output_dir, filename, precision, create_folder=True)
        elif format == 'tsv':
            MatrixExporter.to_tsv(self.matrix, self.row_sentence, self.col_sentence, output_dir, filename, precision, create_folder=True)
        elif format == 'xlsx':
            MatrixExporter.to_xlsx(self.matrix, self.row_sentence, self.col_sentence, output_dir, filename, precision, create_folder=True)
        else:
            raise ValueError(f"Format non supporté: {format}")

    def __repr__(self) -> str:
        return (f"<AttentionMatrix L={self.layer_id} H={self.head_id} "
                f"Rows={len(self.row_sentence.tokens)} Cols={len(self.col_sentence.tokens)}>")