from __future__ import annotations

import argparse
from argparse import Namespace


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Análisis de hashtags sobre tweets.")
    parser.add_argument(
        "--loader",
        choices=("kaggle", "json", "csv", "rapidapi"),
        default="csv",
        help=(
            "Origen de datos: csv (local), kaggle (remoto), "
            "rapidapi (Twitter), json (no disponible)."
        ),
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
    parser.add_argument(
        "--rapidapi-tweet-count",
        type=int,
        default=300,
        metavar="N",
        help="Número de tweets a pedir con --loader rapidapi (mínimo 1).",
    )
    parser.add_argument(
        "--use-file",
        action="store_true",
        help=(
            "Con --loader rapidapi, usa data/rapidapi_tweets.csv si existe en lugar "
            "de llamar a la API."
        ),
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="Construye el grafo de menciones y calcula métricas (NetworkX).",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help=(
            "Genera un prompt desde la red y abre chat con el LLM local "
            "(requiere --network o lo ejecuta antes)."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default="google/gemma-4-E2B-it",
        help="Identificador Hugging Face del modelo causal (por defecto Gemma 4 E2B-it).",
    )
    parser.add_argument(
        "--no-network-plot",
        action="store_true",
        help="No mostrar la figura del grafo al analizar la red.",
    )
    return parser


def parse_cli_args(argv: list[str] | None = None) -> Namespace:
    return build_argument_parser().parse_args(argv)
