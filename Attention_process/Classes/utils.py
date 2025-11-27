# attention_lib/utils.py
from pathlib import Path
from typing import List

def get_documents_heads(file: Path) -> List[int]:
    """
    Lit un fichier contenant les têtes de documents et retourne une liste d'entiers.
    
    Args:
        file (Path): Chemin vers le fichier des têtes de documents.
    
    Returns:
        List[int]: Liste des têtes de documents.
    """
    if not file.exists():
        # Gestion basique d'erreur si le fichier n'existe pas lors de l'import
        return []
        
    with file.open("r", encoding="utf-8") as f:
        heads = [int(line.strip()) for line in f if line.strip().isdigit()]
    return heads