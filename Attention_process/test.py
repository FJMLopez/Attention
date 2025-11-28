# main.py
from Attention_process.io_outils.utils import load_json_file
from Attention_process.io_outils.concat_loader import ConcatModelLoader
from Attention_process.utils import setup_logging
from Attention_process.auto_trace import apply_tracing

import logging

def process_file(json_path: str):
    # 1. Lecture brute
    raw_data = load_json_file(json_path)
    
    # 2. Transformation en objets métier
    # ConcatModelLoader s'occupe de tout le parsing complexe (segments, eos, slicing numpy)
    attention_matrices = ConcatModelLoader.load_from_json_data(raw_data)
    
    print(f"Extraction terminée : {len(attention_matrices)} matrices trouvées.")
    
    if not attention_matrices:
        return

    # 3. Traitement standard (le reste de votre pipeline ne change pas)
    for att_mat in attention_matrices:
        # Exemple : Fusion BPE
        processed_att = att_mat.merge_bpe(method='max')
        
        # Exemple : Sauvegarde
        processed_att.save(
            output_dir="./output_concat", 
            format="json",
            filename_suffix=f"_L{att_mat.layer_id}_H{att_mat.head_id}"
        )


# Simulation d'appel
if __name__ == "__main__":

    # Active le logger
    setup_logging(level=logging.DEBUG)
    logger = logging.getLogger("Attention_process")

    # Active le tracing automatique de TOUTES les fonctions de ce fichier
    apply_tracing(globals())
    logger.info("Démarrage du traitement du fichier de test.") 

    test_folder = "188.json"
    file = f"/home/getalp/lopezfab/temp/temp/test_attn/{test_folder}"

    print(process_file(file))

