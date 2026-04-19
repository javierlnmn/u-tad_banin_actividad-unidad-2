from __future__ import annotations

import argparse
from argparse import Namespace


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Análisis de hashtags sobre tweets.")
    parser.add_argument(
        "--loader",
        choices=("kaggle", "json", "csv"),
        default="csv",
        help="Origen de datos: csv (local), kaggle (remoto), json (no disponible).",
    )
    parser.add_argument(
        "--csv-path",
        default="data/Bitcoin_tweets_dataset_2.csv",
        help="Ruta al CSV local (solo con --loader csv).",
    )
    parser.add_argument(
        "--kaggle-dataset",
        default="kaushiksuresh147/bitcoin-tweets",
        help="Slug del dataset en Kaggle (solo con --loader kaggle).",
    )
    parser.add_argument(
        "--kaggle-file",
        default="Bitcoin_tweets_dataset_2.csv",
        help="Nombre del fichero dentro del dataset (solo con --loader kaggle).",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Guardar cleaned_dataset.csv en --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Carpeta de salida para --export (cleaned_dataset.csv).",
    )
    return parser


def parse_cli_args(argv: list[str] | None = None) -> Namespace:
    return build_argument_parser().parse_args(argv)
