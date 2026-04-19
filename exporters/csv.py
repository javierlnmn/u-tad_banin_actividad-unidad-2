from __future__ import annotations

from exporters.base import DataExporter


class CSVExporter(DataExporter):
    def export(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(
            self.output_dir / "cleaned_dataset.csv", index=False, encoding="utf-8"
        )
