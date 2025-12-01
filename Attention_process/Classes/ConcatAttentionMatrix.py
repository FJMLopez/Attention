# attention_lib/models/concat_attention_matrix.py
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Union

from Attention_process.Classes.Matrix import Matrix
from Attention_process.Classes.Sentence import Sentence
from Attention_process.Classes.AttentionMatrix import AttentionMatrix
from Attention_process.services.MatrixExporter import MatrixExporter

import logging
logger = logging.getLogger(__name__)

@dataclass
class ConcatAttentionMatrix:
    """
    Représente une matrice d'attention pour un modèle par concaténation.
    
    Structure :
    - Lignes (Query) : La phrase courante (Target).
    - Colonnes (Keys) : Une concaténation de plusieurs phrases de contexte [+ OPTIONNELLEMENT la phrase courante].
    
    Cette classe permet de manipuler l'ensemble comme une seule matrice pour la performance,
    tout en gardant la séparation sémantique des phrases de contexte.
    """
    layer_id: int
    head_id: int
    matrix: Matrix
    row_sentence: Sentence              # Phrase courante
    context_sentences: List[Sentence]   # Liste ordonnée des phrases de contexte
    source_model: Optional[str] = None
    
    # Cache pour les offsets des contextes (optimisation)
    _ctx_offsets: List[int] = field(init=False, repr=False)
    # Propriété pour savoir si la matrice inclut la partie "Current->Current"
    _has_self_part: bool = field(init=False, repr=False)

    def __post_init__(self):
        n_rows = len(self.row_sentence.tokens)
        
        # Longueur des contextes seuls
        ctx_lengths = [len(s.tokens) for s in self.context_sentences]
        len_contexts = sum(ctx_lengths)
        
        # Dimensions réelles de la matrice
        mat_rows, mat_cols = self.matrix.shape

        # Cas 1 : Contexte uniquement (Classique)
        if mat_cols == len_contexts:
            self._has_self_part = False
        # Cas 2 : Contexte + Phrase Courante (Nouveau)
        elif mat_cols == len_contexts + n_rows:
            self._has_self_part = True
        else:
            raise ValueError(
                f"Incohérence dimensions [L{self.layer_id}H{self.head_id}]:\n"
                f"Matrice {self.matrix.shape}\n"
                f"Attendus : {n_rows} x {len_contexts} (Ctx seul) "
                f"OU {n_rows} x {len_contexts + n_rows} (Ctx + Self)"
            )
        
        # Offsets : on ajoute celui de la partie 'self' si présente
        self._ctx_offsets = [0] + list(np.cumsum(ctx_lengths[:-1]))
    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    # --- Méthodes Métier Spécifiques ---

    def get_attention_for_context(self, context_index: int) -> AttentionMatrix:
        """
        Extrait une sous-matrice standard (AttentionMatrix) correspondant 
        à l'attention portée sur UNE phrase de contexte spécifique.
        
        Args:
            context_index (int): Index de la phrase de contexte dans la liste context_sentences.
            
        Returns:
            AttentionMatrix: La vue locale (Phrase Courante vs Contexte[i]).
        """
        if not (0 <= context_index < len(self.context_sentences)):
            raise IndexError(f"Index de contexte invalide : {context_index}")

        ctx_snt = self.context_sentences[context_index]
        start_col = self._ctx_offsets[context_index]
        end_col = start_col + len(ctx_snt.tokens)

        # Extraction purement numpy (pas de copie si possible selon l'implémentation Matrix)
        # On suppose que Matrix.data est accessible, sinon il faut ajouter une méthode slice à Matrix.
        # Ici on utilise une méthode hypothétique de Matrix ou on accède à data
        sub_data = self.matrix.data[:, start_col:end_col]
        
        return AttentionMatrix(
            layer_id=self.layer_id,
            head_id=self.head_id,
            matrix=Matrix(sub_data),
            row_sentence=copy.deepcopy(self.row_sentence),
            col_sentence=copy.deepcopy(ctx_snt),
            source_model=f"{self.source_model}_Ctx{context_index}"
        )

    def merge_bpe(self, 
                  method: Literal['max', 'mean', 'sum'] = 'max',
                  bpe_mark: str = '@@') -> 'ConcatAttentionMatrix':
        """
        Applique la fusion BPE sur la phrase courante et SUR TOUTES les phrases de contexte.
        Gère le décalage des indices colonnes correctement.
        """
        # 1. Groupes Lignes (Phrase courante)
        row_groups = Sentence.list_fusion_bpe(self.row_sentence.tokens, BPE_mark=bpe_mark)

        # 2. Groupes Colonnes (Global)
        # Il faut calculer les groupes pour chaque phrase de contexte, 
        # puis décaler les indices pour qu'ils correspondent à la grande matrice concaténée.
        global_col_groups = []
        current_offset = 0
        new_context_sentences = []

        for ctx_snt in self.context_sentences:
            # Groupes locaux (ex: [[0,1], [2]])
            local_groups = Sentence.list_fusion_bpe(ctx_snt.tokens, BPE_mark=bpe_mark)
            
            # Ajout au global avec décalage (ex: [[10,11], [12]])
            for grp in local_groups:
                global_col_groups.append([idx + current_offset for idx in grp])
            
            current_offset += len(ctx_snt.tokens)
            
            # Préparation des nouvelles phrases (sémantique)
            new_snt = copy.deepcopy(ctx_snt)
            new_snt.fusion_bpe(list_groupes_bpe=local_groups, BPE_mark=bpe_mark)
            new_context_sentences.append(new_snt)

        # 2.1. Traitement de la partie Self (si présente dans les colonnes)
        if self._has_self_part:
            # On réutilise les row_groups calculés en 1, mais décalés
            for grp in row_groups:
                global_col_groups.append([idx + current_offset for idx in grp])

        # 3. Réduction Matrice (Une seule grosse opération)
        new_matrix = self.matrix.merge_bpe(row_groups, global_col_groups, method=method)

        # 4. Mise à jour Phrase Courante
        new_row_snt = copy.deepcopy(self.row_sentence)
        new_row_snt.fusion_bpe(list_groupes_bpe=row_groups, BPE_mark=bpe_mark)

        return ConcatAttentionMatrix(
            layer_id=self.layer_id,
            head_id=self.head_id,
            matrix=new_matrix,
            row_sentence=new_row_snt,
            context_sentences=new_context_sentences,
            source_model=self.source_model
        )
    
    def remove_padding(self, padding_mark: str = "<pad>") -> 'ConcatAttentionMatrix':
        """Supprime le padding sur la ligne et sur l'ensemble des contextes."""
        
        # 1. Indices Lignes
        row_pad_indices = Sentence.list_suppr_pad(self.row_sentence.tokens, padding_mark)
        
        # 2. Indices Colonnes Globaux
        global_col_pad_indices = []
        current_offset = 0
        new_context_sentences = []

        for ctx_snt in self.context_sentences:
            local_pads = Sentence.list_suppr_pad(ctx_snt.tokens, padding_mark)
            # Décalage
            global_col_pad_indices.extend([idx + current_offset for idx in local_pads])
            
            current_offset += len(ctx_snt.tokens)
            
            # Mise à jour phrases
            new_snt = copy.deepcopy(ctx_snt)
            new_snt.suppr_pad(list_index=local_pads)
            new_context_sentences.append(new_snt)

        # 2.1 Self (si présent)
        if self._has_self_part:
            global_col_pad_indices.extend([idx + current_offset for idx in row_pad_indices])

        # 3. Opération Matrice
        new_matrix = self.matrix.remove_indices(row_indices=row_pad_indices, col_indices=global_col_pad_indices)
        
        # 4. Mise à jour Ligne
        new_row_snt = copy.deepcopy(self.row_sentence)
        new_row_snt.suppr_pad(list_index=row_pad_indices)

        return ConcatAttentionMatrix(
            layer_id=self.layer_id,
            head_id=self.head_id,
            matrix=new_matrix,
            row_sentence=new_row_snt,
            context_sentences=new_context_sentences,
            source_model=self.source_model
        )

    def normalize(self, method: Literal["minmax", "max", "row_stochastic"] = "minmax") -> 'ConcatAttentionMatrix':
        """Normalise la matrice globale."""
        return ConcatAttentionMatrix(
            layer_id=self.layer_id,
            head_id=self.head_id,
            matrix=self.matrix.normalize(method),
            row_sentence=copy.deepcopy(self.row_sentence),
            context_sentences=copy.deepcopy(self.context_sentences),
            source_model=self.source_model
        )

    def threshold(self, 
                  method: Literal["uniform", "value", "to_uniform"] = "uniform", 
                  value: Optional[float] = None,
                ) -> 'ConcatAttentionMatrix':
        """
        Applique un seuillage sur la matrice.

        Args:
            method: Méthode de seuillage ('uniform', 'value', 'to_uniform').
            value: Valeur explicite du seuil si method='value'.
            total_seq_len: (Optionnel) Longueur totale de la séquence (Contextes + Query) 
                           pour calculer l'uniforme correct (1 / total_seq_len).
                           Si None et method='uniform', utilise le nombre de colonnes (Contexte seul).
        """
        
        # Logique spécifique : Uniforme basé sur une longueur externe
        if method == "uniform" :
            # On détourne la méthode 'value' de la classe Matrix qui fait seuil = 1.0 / value
            print(f"len(ctx): {[len(contexte) for contexte in self.context_sentences]}")
            print(f"len(crt): {len(self.row_sentence.tokens)}")
            print(f"total_seq_len: {sum(len(contexte) for contexte in self.context_sentences) + len(self.row_sentence.tokens)}")
            logger.debug(f"ConcatAttentionMatrix L{self.layer_id}H{self.head_id}: Seuillage uniforme avec total_seq_len={sum(len(contexte) for contexte in self.context_sentences) + len(self.row_sentence.tokens)}.")
            new_matrix = self.matrix.threshold(method="value", value=sum(len(contexte) for contexte in self.context_sentences) + len(self.row_sentence.tokens))
        else:
            # Comportement standard (basé sur value ou sur self.cols)
            new_matrix = self.matrix.threshold(method, value)

        return ConcatAttentionMatrix(
            layer_id=self.layer_id,
            head_id=self.head_id,
            matrix=new_matrix,
            row_sentence=copy.deepcopy(self.row_sentence),
            context_sentences=copy.deepcopy(self.context_sentences),
            source_model=self.source_model
        )

    # --- Export ---

    def save(self, 
             output_dir: str, 
             format: Literal['json', 'pdf_heatmap', 'tsv', 'xlsx'] = 'json',
             precision: int = 2,
             filename_suffix: str = "") -> None:
        """
        Sauvegarde. 
        Note : Pour simplifier l'export, on crée une 'fausse' phrase globale concaténée pour le contexte,
        afin d'utiliser le MatrixExporter standard.
        """
        # 1. Concaténation virtuelle des tokens de contexte pour l'affichage
        full_ctx_tokens = []
        for s in self.context_sentences:
            full_ctx_tokens.extend(s.tokens)

        # 2. Ajout des tokens de la phrase courante (SI PRÉSENTS dans la matrice)
        if self._has_self_part:
            full_ctx_tokens.extend(self.row_sentence.tokens)
            
            
        global_ctx_snt = Sentence(
            tokens=full_ctx_tokens,
            system_id=self.context_sentences[0].system_id if self.context_sentences else -1
        )
        
        model_part = f"_{self.source_model}" if self.source_model else ""
        filename = f"L{self.layer_id}_H{self.head_id}{model_part}{filename_suffix}"

        if format == 'json':
            MatrixExporter.to_json(self.matrix, self.row_sentence, global_ctx_snt, output_dir, filename, precision, create_folder=True)
        elif format == 'tsv':
            MatrixExporter.to_tsv(self.matrix, self.row_sentence, global_ctx_snt, output_dir, filename, precision, create_folder=True)
        elif format == 'xlsx':
            MatrixExporter.to_xlsx(self.matrix, self.row_sentence, global_ctx_snt, output_dir, filename, precision, create_folder=True)
        elif format == 'pdf_heatmap':
            MatrixExporter.to_pdf_heatmap(self.matrix, self.row_sentence, global_ctx_snt, output_dir, filename, create_folder=True)
        else:
            raise ValueError(f"Format non supporté: {format}")




if __name__ == "__main__":
    # Exemple d'utilisation basique
    row_snt = Sentence(tokens=["I", "love", "programming", "@@", "in", "Python", "<pad>"], system_id=0)
    ctx1 = Sentence(tokens=["Python", "is", "a", "great", "language", "<pad>"], system_id=1)
    ctx2 = Sentence(tokens=["I", "enjoy", "solving", "problems", "<pad>", "<pad>"], system_id=2)

    data = np.random.rand(7, 12)  # 7 tokens row, 6 + 6 tokens context
    mat = Matrix(data)

    concat_att_mat = ConcatAttentionMatrix(
        layer_id=0,
        head_id=0,
        matrix=mat,
        row_sentence=row_snt,
        context_sentences=[ctx1, ctx2],
        source_model="ConcatDemo"
    )

    print("Shape initiale:", concat_att_mat.shape)

    merged = concat_att_mat.merge_bpe(method='max', bpe_mark='@@')
    print("Shape après fusion BPE:", merged.shape)

    no_pad = merged.remove_padding(padding_mark="<pad>")
    print("Shape après suppression padding:", no_pad.shape)

    no_pad.save(output_dir="./output_demo", format="json", filename_suffix="_demo")