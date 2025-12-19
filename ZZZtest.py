# main.py
from Attention_process.io_outils.multienc_loader import MultiEncoderLoader
from Attention_process.io_outils.utils import load_json_file
import logging
import numpy as np
import sys

logger = logging.getLogger("Attention_process")

def process_multi_encoder(json_path: str):
    raw_data = load_json_file(json_path)
    
    # Chargement
    data_dict = MultiEncoderLoader.load_from_json_data(raw_data)
    wl_matrices = data_dict['word_level']
    sl_matrices = data_dict['sentence_level']
    with np.printoptions(threshold=sys.maxsize):
        print(len(data_dict['word_level'][0].col_sentence))
        print(f"row len: {len(wl_matrices[0].row_sentence)}")
        print(f"col len: {len(wl_matrices[0].col_sentence)}")
        print(f"Matrice shape: {wl_matrices[0].shape}")
        print(wl_matrices[0].matrix[0,-25:])

    logger.info(f"Chargé : {len(wl_matrices)} matrices Word-Level, {len(sl_matrices)} matrices Sentence-Level")

    # 1. Traitement Word-Level (Mot à Mot)
    # for mat in wl_matrices:
    mat = wl_matrices[0]
    # mat représente l'attention de Crt vers Ctx_k
    mat = mat.remove_padding()
    mat = mat.threshold(method="uniform")
    with np.printoptions(threshold=sys.maxsize):
        print("** uniform **")
        print(len(mat.col_sentence))
        print(mat.shape)
        print(mat.matrix[0,:])
        print(mat.col_sentence)
    mat = mat.merge_bpe()
    with np.printoptions(threshold=sys.maxsize):
        print("** merge_bpe **")
        print(mat.col_sentence)
        print(len(mat.col_sentence))
        print(mat.shape)
        print(mat.matrix[0,:])
    mat = mat.normalize("minmax")
    with np.printoptions(threshold=sys.maxsize):
        print("** normalize **")
        print(len(mat.col_sentence))
        print(mat.shape)
        print(mat.matrix[0,:])
    processed = (
    mat
    )
    processed.save("./output_multi/wl", format="pdf_heatmap")
    # 2. Traitement Sentence-Level (Hiérarchique)
    for mat in sl_matrices:
        # mat représente l'attention de Crt vers [Ctx0, Ctx1...]
        # Ici, l'axe X (colonnes) n'est pas des mots, mais des identifiants de phrases.
        # merge_bpe sur les colonnes n'a pas de sens (ce ne sont pas des BPE), 
        # mais merge_bpe sur les lignes (Crt) fonctionne.
        
        # Attention : Sentence.list_fusion_bpe va voir "Context_0", pas de "@@", donc pas de fusion colonnes. C'est parfait.
        
        processed_sl = (
            mat
            .merge_bpe()     # Fusionnera les lignes (Crt) uniquement
            .remove_padding() # Supprimera le padding des lignes (Crt)
            .normalize("row_stochastic") # Pour voir la distribution de proba sur les phrases
        )
        processed_sl.save("./output_multi/sl", format="pdf_heatmap", filename_suffix="_Hierarchical")

process_multi_encoder("/home/getalp/lopezfab/temp/temp/temp/han_attn2/1.json")