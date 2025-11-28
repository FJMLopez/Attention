# attention_lib/io/utils.py
import json
import logging
from pathlib import Path
from typing import Dict, Any

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Charge un fichier JSON."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")
    
    with path.open('r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Erreur de décodage JSON dans {file_path}: {e}")

def check_create_dir(dir_path: str):
    """Crée le dossier s'il n'existe pas."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)