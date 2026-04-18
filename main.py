from __future__ import annotations

from cli import parse_cli_args
from loaders.utils import build_data_loader
from extractor import DataExtractor


if __name__ == "__main__":
    ns = parse_cli_args()

    loader = build_data_loader(
        loader_name=ns.loader,
        csv_path=ns.csv_path,
        kaggle_dataset=ns.kaggle_dataset,
        kaggle_file=ns.kaggle_file,
    )

    DataExtractor(loader=loader).generate_hashtag_wordcloud()
