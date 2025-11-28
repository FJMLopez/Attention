import json
import numpy as np
from pathlib import Path
from typing import Any
from Attention_process.Classes.Matrix import Matrix

# Gestion des dépendances optionnelles (xlsxwriter)
import xlsxwriter

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