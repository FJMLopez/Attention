# src/config.py
import argparse
from pathlib import Path
from src.types import AppConfig


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description='NMT-ParCorFull Alignment & Analysis')
    parser.add_argument('--corpus-source', required=True, type=Path, help='Source corpus file')
    parser.add_argument('--corpus-target', required=True, type=Path, help='Target corpus file')
    parser.add_argument('--system-data', required=True, type=Path, help='System attention data')
    parser.add_argument('--parcor-path', required=True, type=Path, help='Base path to ParCorFull2 data')
    parser.add_argument('--evaluate-language', choices=['source', 'target'], default='source')
    parser.add_argument('--canmt-system', choices=['concat', 'multienc'], default='concat')
    parser.add_argument('--output-file', type=Path, default=Path("./attention_analysis.results"))
    parser.add_argument('--threshold', type=float, default=0.5, help='WER Threshold')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log-file', type=Path, default=None, help='Path to save execution logs')
    
    args = parser.parse_args()
    
    return AppConfig(
        corpus_source=args.corpus_source,
        corpus_target=args.corpus_target,
        system_data=args.system_data,
        evaluate_language=args.evaluate_language,
        canmt_system=args.canmt_system,
        output_file=args.output_file,
        parcor_base_path=args.parcor_path,
        wer_threshold=1000.0 if args.evaluate_language == 'target' else args.threshold,
        verbose=args.verbose,
        debug=args.debug,
        log_file=args.log_file,
    )