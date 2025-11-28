import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, List
from Attention_process.Classes.Matrix import Matrix
import logging

# Gestion des dépendances optionnelles (xlsxwriter)
import xlsxwriter
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

class MatrixExporter:
    """
    Classe utilitaire responsable de l'exportation des matrices.
    Sépare la logique I/O de la logique mathématique.
    """

    @staticmethod
    def _ensure_folder(path: str, create: bool = False):
        p = Path(path)
        if create:
            p.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            raise FileNotFoundError(f"Le dossier {path} n'existe pas.")

    @staticmethod
    def _create_dataframe(matrix_data: List[List[float]], 
                          row_tokens: List[str], 
                          col_tokens: List[str]) -> 'pd.DataFrame':
        """Helper pour créer un DataFrame Pandas proprement."""
        
        df = pd.DataFrame(matrix_data, columns=col_tokens)
        # On ajoute les row_tokens comme une colonne temporaire ou un index
        # Pour coller à votre logique 'visualize' qui attend la colonne 0 comme index
        df.insert(0, "Current_Sentence_Tokens", row_tokens)
        return df
    
    # --- Exports Texte / Données ---

    @staticmethod
    def to_xlsx(matrix: Matrix, crt: Any, ctx: Any, folder: str, filename: str, precision: int = 2, create_folder: bool = False):
        """Exporte en Excel avec mise en forme conditionnelle."""

        MatrixExporter._ensure_folder(folder, create_folder)
        full_path = Path(folder) / f"{filename}.xlsx"

        workbook = xlsxwriter.Workbook(str(full_path))
        worksheet = workbook.add_worksheet()
        
        # Formats
        fmt_highlight = workbook.add_format({'bg_color': 'cyan', 'bold': True})
        
        # Header (Phrase Contexte)
        worksheet.write(0, 0, f"{crt.system_id}-{int(crt.system_id) - int(ctx.system_id)}")
        
        # On suppose que crt et ctx sont des objets Sentence (list-like pour les tokens)
        for col_idx, tok in enumerate(ctx.tokens, start=1):
            worksheet.write(0, col_idx, tok)
            
        # Rows (Phrase Courante + Data)
        rows, cols = matrix.shape
        data = matrix.data
        
        for i in range(rows):
            # Token ligne
            if i < len(crt.tokens):
                worksheet.write(i + 1, 0, crt.tokens[i])
            
            row_max = np.max(data[i])
            
            for j in range(cols):
                val = data[i, j]
                val_str = f"{val:.{precision}f}"
                
                if val == 0:
                     worksheet.write(i + 1, j + 1, ".")
                elif val == row_max and val > 0:
                    worksheet.write(i + 1, j + 1, val_str, fmt_highlight)
                else:
                    worksheet.write(i + 1, j + 1, val_str)
                    
        workbook.close()

    @staticmethod
    def to_tsv(matrix: Matrix, crt: Any, ctx: Any, folder: str, filename: str, precision: int = 2, create_folder: bool = False):
        """Exporte en TSV."""
        MatrixExporter._ensure_folder(folder, create_folder)
        full_path = Path(folder) / f"{filename}.tsv"

        rows, cols = matrix.shape
        data = matrix.data

        with open(full_path, "w", encoding="utf-8") as f:
            # Header
            header_meta = f"{crt.system_id}-{int(crt.system_id) - int(ctx.system_id)}"
            header_toks = "\t".join(ctx.tokens)
            f.write(f"{header_meta}\t{header_toks}\n")

            for i in range(rows):
                tok = crt.tokens[i] if i < len(crt.tokens) else ""
                # Formatage des valeurs
                vals = [f"{v:.{precision}f}" for v in data[i]]
                line = f"{tok}\t" + "\t".join(vals)
                f.write(line + "\n")

    @staticmethod
    def to_json(matrix: Matrix, crt: Any, ctx: Any, folder: str, filename: str, precision: int = 2, create_folder: bool = False):
        """Exporte en JSON."""
        MatrixExporter._ensure_folder(folder, create_folder)
        full_path = Path(folder) / f"{filename}.json"

        # On suppose que crt et ctx ont une méthode toJSON ou sont convertibles
        # Ici on utilise __dict__ par défaut si disponible
        def serialize(obj):
            return obj.__dict__ if hasattr(obj, '__dict__') else str(obj)

        output = {
            'crt': json.loads(crt.toJSON()) if hasattr(crt, 'toJSON') else str(crt),
            'ctx': json.loads(ctx.toJSON()) if hasattr(ctx, 'toJSON') else str(ctx),
            'matrix': matrix.to_list(precision)
        }

        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)


            # --- Visualisation PDF / Heatmap ---

    @staticmethod
    def to_pdf_heatmap(matrix: Any, crt: Any, ctx: Any, folder: str, filename: str, create_folder: bool = False):
        """
        Génère une heatmap Seaborn et la sauvegarde en PDF.
        Cases agrandies pour une meilleure lisibilité.
        """

        MatrixExporter._ensure_folder(folder, create_folder)
        full_path = Path(folder) / f"{filename}.pdf"

        # 1. Préparation des données
        data_array = matrix.data
        row_labels = crt.tokens
        col_labels = ctx.tokens

        df = pd.DataFrame(data_array, columns=col_labels)
        df.insert(0, "Row_Labels", row_labels)

        # ---------------------------------------------------------
        # 2. Configuration de la taille (MODIFIÉE POUR AGRANDIR)
        # ---------------------------------------------------------
        n_rows = len(df)
        n_cols = len(df.columns) - 1 # -1 car la col 0 est les labels

        # Définition de la taille par cellule (en pouces)
        # Augmentez ces valeurs si vous voulez des cases encore plus grosses
        cell_width_inch = 0.8 
        cell_height_inch = 0.6

        # Calcul de la taille totale (avec un minimum de sécurité pour les marges)
        fig_width = max(10, n_cols * cell_width_inch)
        fig_height = max(8, n_rows * cell_height_inch)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # ---------------------------------------------------------

        numeric_data = df.iloc[:, 1:].astype(float)
        
        # Masque et Labels
        epsilon = 1e-6
        mask = numeric_data <= epsilon 
        annot_labels = numeric_data.map(lambda x: f"{x:.2f}" if x > epsilon else "")

        # 3. Tracé Heatmap
        sns.heatmap(
            numeric_data, 
            vmin=0, vmax=1, 
            cmap=sns.cm.rocket_r,
            annot=annot_labels,
            fmt="",
            mask=mask,           
            linecolor='gray',
            linewidths=.5,
            ax=ax,
            # Ajout : Contrôle de la taille du texte dans les cases
            annot_kws={"size": 10} 
        )

        # 4. Configuration Axes
        # Augmentation de la taille de police des axes aussi
        ax.set_yticklabels(df["Row_Labels"], rotation=0, fontsize=12)
        ax.set_xticklabels(col_labels, rotation=70, fontsize=12)
        
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        
        # Labels des axes (Titres)
        ax.set_xlabel("Context (Keys)", fontsize=14, labelpad=15)
        ax.set_ylabel("Current Sentence (Queries)", fontsize=14, labelpad=15)
        
        plt.tight_layout()

        try:
            plt.savefig(full_path, format='pdf', bbox_inches='tight')
            logger.debug(f"PDF Heatmap sauvegardé : {full_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde PDF : {e}")
        finally:
            plt.close(fig)






















