from __future__ import annotations

import logging
import sys

import pandas as pd
from dotenv import load_dotenv

from cli import parse_cli_args
from exporters.csv import CSVExporter
from extractor import DataExtractor
from loaders.utils import Loaders, build_data_loader

load_dotenv()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    ns = parse_cli_args()
    try:
        loader = build_data_loader(
            loader_name=Loaders(ns.loader),
            csv_path=ns.csv_path,
            kaggle_dataset=ns.kaggle_dataset,
            kaggle_file=ns.kaggle_file,
            rapidapi_tweet_count=max(1, ns.rapidapi_tweet_count),
            rapidapi_use_file=ns.use_file,
        )
    except Exception as err:
        print(f"Error al cargar los datos: {err}", file=sys.stderr)
        sys.exit(2)

    if Loaders(ns.loader) in (Loaders.KAGGLE, Loaders.CSV):
        extractor = DataExtractor(loader=loader)
        stats = extractor.analytics_hashtags_extended()

        if ns.export:
            df = extractor.data.copy()
            df["text"] = df["text"].apply(extractor.clean_text)
            df["hashtags"] = df["text"].apply(extractor.extract_hashtags)
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
            CSVExporter(df, ns.output_dir).export()

        extractor.generate_hashtag_wordcloud(overall_df=stats["overall"])

    if Loaders(ns.loader) == Loaders.RAPIDAPI:
        print(loader.load())
