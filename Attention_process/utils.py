# utils.py
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(log_file: str = "app.log", level: int = logging.INFO) -> logging.Logger:
    """
    Configure le système de logging pour l'application.
    
    Affiche les logs dans la console (stdout) et les sauvegarde dans un fichier
    avec rotation (pour éviter que le fichier ne devienne trop gros).

    Args:
        log_file (str): Chemin du fichier de log.
        level (int): Niveau de logging (ex: logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: L'objet logger configuré.
    """
    # Création du formatteur
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Handler Fichier (Rotation après 500MB, garde 3 backups)
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setFormatter(formatter)

    # Configuration du logger racine
    logger = logging.getLogger("Attention_process")
    logger.setLevel(level)
    
    # Éviter la duplication des logs si la fonction est appelée plusieurs fois
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

