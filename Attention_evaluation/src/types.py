# src/types.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union

# --- Type Aliases ---
# Représente une opération d'alignement : (Nom, Index Ref, Index Hyp)
# Exemple : ('match', 0, 0) ou ('ins', None, 1)
AlignmentOp = Tuple[str, Optional[int], Optional[int]]

@dataclass
class AppConfig:
    """
    Configuration globale de l'application.
    Contient tous les arguments CLI et les constantes de réglage.
    """
    # Chemins de fichiers (Paths)
    corpus_source: Path
    corpus_target: Path
    system_data: Path
    parcor_base_path: Path
    output_file: Path
    log_file: Optional[Path] = None 

    # Paramètres d'exécution
    evaluate_language: str = 'source'  # 'source' ou 'target'
    canmt_system: str = 'concat'       # 'concat' ou 'multienc'
    
    # Paramètres algorithmiques
    wer_threshold: float = 0.5
    coref_link_score_mode: str = 'max' # 'max' ou 'avg'
    
    # Options de débogage
    verbose: bool = False
    debug: bool = False


@dataclass
class Mention:
    """
    Représente une mention de coréférence extraite des fichiers XML ParCorFull.
    """
    id: str
    fileid: str
    span: str
    coref_class: str          # L'ID du cluster de coréférence (ex: "set_10")
    mention_text: str
    word_indices: List[int]   # Liste des IDs de mots (ex: [1, 2, 3])


@dataclass
class CorpusSentence:
    """
    Représente une phrase du corpus alignée et annotée.
    """
    raw_text: str             # Texte brut du fichier XML
    tokenized_text: str       # Texte après tokenization.py
    alignment: List[AlignmentOp] # Liste des opérations d'édition (Raw -> Tok)
    word_ids: List[str]       # IDs uniques des mots dans le corpus
    annotated_text: str       # Texte décoré (ex: "#[chat]#-set_1")


@dataclass
class CorpusData:
    """
    Conteneur pour tout le corpus chargé (DiscoMT ou News).
    """
    # Map: texte brut -> (texte tokenisé, alignement)
    text_map: Dict[str, Tuple[str, List[AlignmentOp]]] 
    
    # Map: 'prefix-word_id' -> 'Mot textuel'
    words: Dict[str, str]
    
    # Map: 'prefix-word_id' -> 'set_id' (seulement pour les mots dans une mention)
    words_in_coref: Dict[str, str]
    
    # Liste plate de toutes les mentions trouvées
    coref_mentions: List[Mention]


@dataclass
class SystemSequence:
    """
    Représente une entrée dans le fichier de sortie du système (Attention).
    """
    current: str                # La phrase générée/analysée
    context: str                # La phrase de contexte
    attention: List[List[float]] # Matrice d'attention [len(current) x len(context)]


@dataclass
class AnalysisResult:
    """
    Résultat de l'évaluation d'une séquence.
    """
    seq_id: str
    current_sent: str
    context_sent: str
    
    # Liste des métriques pour chaque lien de coréférence trouvé.
    # Format d'un item : [IsAntecedent(bool), HasWeight(bool), Unused(bool), Score(float)]
    metrics: List[List[Union[bool, float]]] = field(default_factory=list)