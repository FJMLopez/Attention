import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(verbose: bool = False, debug: bool = False, log_file: Optional[Path] = None) -> None:
    """
    Configure le logger racine.
    
    Args:
        verbose: Niveau INFO.
        debug: Niveau DEBUG.
        log_file: Chemin optionnel vers un fichier de log.
    """
    
    # Détermination du niveau
    level = logging.WARNING
    if verbose:
        level = logging.INFO
    if debug:
        level = logging.DEBUG

    # Format des logs
    log_fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Récupération du logger racine
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Nettoyage des handlers existants (pour éviter les doublons si rappelé)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 1. Handler Console (Stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(log_fmt)
    root_logger.addHandler(console_handler)

    # 2. Handler Fichier (Optionnel)
    # if log_file:
    # On s'assure que le dossier parent existe
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # mode='w' écrase le fichier à chaque lancement. 
    # Utilisez mode='a' pour ajouter à la suite.
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(log_fmt)
    root_logger.addHandler(file_handler)
    
    # Petit message pour confirmer que ça écrit
    logging.debug(f"Logging to file enabled: {log_file}")

    # Faire taire les bibliothèques bruyantes si pas en debug
    if not debug:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)