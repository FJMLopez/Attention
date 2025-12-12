# io_handlers.py
import json
import csv
import logging
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from Attention_process.Classes.AttentionMatrix import AttentionMatrix
from Attention_process.Classes.Sentence import Sentence

logger = logging.getLogger("AttentionProcessor")

class AttentionLoader(ABC):
    @abstractmethod
    def load(self, source: str) -> list[AttentionMatrix]:
        pass

class CustomFormatLoader(AttentionLoader):
    """Simulation d'un chargeur de matrices asymétriques."""
    def load(self, source: str) -> list[AttentionMatrix]:
        logger.info(f"Chargement simulé depuis {source}")
        
        # Simulation: Phrase courante (Query) vs Contexte (Key)
        row_toks = Sentence(system_id=1, tokens= ["Le", "chat", "##on", "mange"])      # 4 tokens
        col_toks = Sentence(system_id=0, tokens= ["Le", "petit", "chat", "est", "là"]) # 5 tokens
        
        matrices = []
        for layer in range(2):
            # Matrice rectangulaire (4x5)
            mat = np.random.rand(len(row_toks), len(col_toks)).astype(np.float32)
            # Pré-normalisation
            mat = mat / mat.sum(axis=1, keepdims=True)
            
            matrices.append(AttentionMatrix(
                layer_id=layer, 
                head_id=0, 
                row_sentence=row_toks, 
                col_sentence=col_toks, 
                matrix=mat,
                source_model="model-v1"
            ))
        return matrices

class AttentionSaver(ABC):
    @abstractmethod
    def save(self, data: AttentionMatrix, output_path: str):
        pass

class JsonSaver(AttentionSaver):
    def save(self, data: AttentionMatrix, output_path: str):
        output = {
            "metadata": {
                "layer": data.layer_id,
                "head": data.head_id,
                "shape": data.shape
            },
            "row_tokens": data.row_sentence, # Phrase Courante
            "col_tokens": data.col_sentence, # Contexte
            "matrix": data.matrix.to_list()
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

class TsvSaver(AttentionSaver):
    def save(self, data: AttentionMatrix, output_path: str):
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            # Header : Vide + Tokens Contexte
            writer.writerow(['ROW\COL'] + data.col_sentence.tokens)
            # Lignes : Token Courant + Poids
            for i, row_vals in enumerate(data.matrix.data):
                writer.writerow([data.row_sentence.tokens[i]] + list(row_vals))

class PdfHeatmapSaver(AttentionSaver):
    def save(self, data: AttentionMatrix, output_path: str):
        try:
            # Taille dynamique selon la taille de la matrice
            h, w = data.matrix.shape
            fig_h = max(6, h * 0.5)
            fig_w = max(8, w * 0.5)
            
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            cax = ax.matshow(data.matrix.data, cmap='viridis', aspect='auto')
            fig.colorbar(cax)

            # Configuration Axe X (Contexte)
            ax.set_xticks(np.arange(len(data.col_sentence)))
            ax.set_xticklabels(data.col_sentence.tokens, rotation=90)
            ax.set_xlabel("Contexte (Keys)")

            # Configuration Axe Y (Phrase Courante)
            ax.set_yticks(np.arange(len(data.row_sentence)))
            ax.set_yticklabels(data.row_sentence.tokens)
            ax.set_ylabel("Phrase Courante (Queries)")

            plt.title(f"Attention L{data.layer_id} H{data.head_id}")
            plt.tight_layout()
            plt.savefig(output_path, format='pdf')
            plt.close()
            logger.info(f"PDF généré: {output_path}")
        except Exception as e:
            logger.error(f"Erreur PDF: {e}", exc_info=True)

class SaverFactory:
    @staticmethod
    def get_saver(format_type: str) -> AttentionSaver:
        match format_type.lower():
            case 'json': return JsonSaver()
            case 'tsv': return TsvSaver()
            case 'pdf': return PdfHeatmapSaver()
            case _: raise ValueError(f"Format inconnu: {format_type}")