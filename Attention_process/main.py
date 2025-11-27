# main.py
import argparse
import logging
from pathlib import Path
from utils import setup_logging
from io_handlers import CustomFormatLoader, SaverFactory
from processors import BPEMerger, AggregationMethod, Normalizer

def main():
    logger = setup_logging("processing.log", level=logging.DEBUG)
    logger.info("--- Démarrage Pipeline Attention (Phrase vs Contexte) ---")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--format", type=str, choices=['json', 'tsv', 'pdf'], default='json')
    parser.add_argument("--merge_bpe", action="store_true")
    parser.add_argument("--agg_method", type=str, choices=['sum', 'mean', 'max'], default='mean')
    parser.add_argument("--normalize", action="store_true")
    
    args = parser.parse_args()

    try:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Chargement
        loader = CustomFormatLoader()
        # Simulation d'input (à remplacer par votre vraie lecture de fichier)
        matrices = loader.load("fake_path.dat")

        if not matrices:
            logger.warning("Aucune donnée à traiter.")
            return

        # 2. Pipeline
        saver = SaverFactory.get_saver(args.format)
        
        for idx, att in enumerate(matrices):
            logger.info(f"Traitement Matrice {idx+1} [L{att.layer_id}-H{att.head_id}]")

            # Fusion BPE
            if args.merge_bpe:
                att = BPEMerger.merge(att, method=AggregationMethod(args.agg_method))

            # Normalisation (Optionnel)
            if args.normalize:
                # Normalisation min-max globale
                att = Normalizer.apply(att, Normalizer.min_max_normalize)
                # Ou normalisation par ligne (Row Stochastic) pour probabilités
                # att = Normalizer.apply(att, Normalizer.row_stochastic)

            # Sauvegarde
            fname = f"L{att.layer_id}_H{att.head_id}_processed.{args.format}"
            saver.save(att, str(out_dir / fname))

        logger.info("Traitement terminé.")

    except Exception as e:
        logger.critical("Erreur fatale:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    import sys # Nécessaire pour sys.exit
    main()