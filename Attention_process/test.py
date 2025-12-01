# main.py
from Attention_process.io_outils.utils import load_json_file
from Attention_process.io_outils.concat_loader import ConcatModelLoader
from Attention_process.utils import setup_logging
from Attention_process.auto_trace import apply_tracing
from Attention_process.Classes.AttentionAggregator import AttentionAggregator

import logging

def process_file(json_path: str):
    # 1. Lecture brute
    raw_data = load_json_file(json_path)
    
    # 2. Transformation en objets métier
    # ConcatModelLoader s'occupe de tout le parsing complexe (segments, eos, slicing numpy)
    attention_matrices = ConcatModelLoader.load_from_json_data(raw_data, include_self_attention=True)
    
    logger.info(f"Extraction terminée : {len(attention_matrices)} matrices trouvées.")
    logger.debug(f"Current sentence len: {len(attention_matrices[0].row_sentence.tokens) if attention_matrices else 'N/A'}")
    logger.debug(f"Context sentences count: {[len(contexte) for contexte in attention_matrices[0].context_sentences] if attention_matrices else 'N/A'}")

    mean_last_layer = AttentionAggregator.aggregate(
        matrices = [mat for mat in attention_matrices if mat.layer_id == max(m.layer_id for m in attention_matrices)],
        method='mean'
    )
    mean_last_layer.merge_bpe(method='max').threshold(method='uniform').save(
        output_dir="./output_concat", 
        format="pdf_heatmap",
        filename_suffix=f"_Mean_Last_Layer"
    )
    # 3. Traitement standard (le reste de votre pipeline ne change pas)
    # for att_mat in attention_matrices:
    #     # Exemple : Fusion BPE
    #     processed_att = att_mat.merge_bpe(method='max')
    #     processed_att = processed_att.threshold(method='uniform')
        
    #     # Exemple : Sauvegarde
    #     processed_att.save(
    #         output_dir="./output_concat", 
    #         format="pdf_heatmap",
    #         filename_suffix=f"_L{att_mat.layer_id}_H{att_mat.head_id}"
    #     )


# Simulation d'appel
if __name__ == "__main__":

    # Active le logger
    setup_logging(level=logging.DEBUG)
    logger = logging.getLogger("Attention_process")

    # Active le tracing automatique de TOUTES les fonctions de ce fichier
    apply_tracing(globals())
    logger.info("Démarrage du traitement du fichier de test.") 

    test_folder = "1850.json"
    file = f"/home/getalp/lopezfab/temp/temp/test_attn/{test_folder}"

    print(process_file(file))

