from __future__ import annotations

import re

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from wordcloud import WordCloud

from loaders.base import DataLoader

DEFAULT_WORDCLOUD_MAX_WORDS = 100


class DataExtractor:
    def __init__(
        self,
        loader: DataLoader,
        chunksize: int = 10_000,
        data: DataFrame | None = None,
    ):
        """
        Inicializa el extractor con el archivo de origen.
        Parámetro:
        source_file: Ruta al archivo de datos (CSV o JSON).
        chunksize: Tamaño de los chunks para el procesamiento.
        loader: Cargador de datos para diferentes formatos.
        data: Si se pasa, se usa ese DataFrame en lugar de volver a cargar desde loader.
        """
        self.loader: DataLoader = loader
        self.chunksize = chunksize
        self.data: DataFrame | None = data if data is not None else loader.load()

    def load_data(self):
        """
        Carga los datos del archivo de origen.
        Implementación esperada:
        - Leer el archivo en el formato correspondiente.
        - Almacenar los datos en self.data.
        """
        self.data = self.loader.load()
        return self.data

    def clean_text(self, text: str) -> str:
        """
        Limpia y normaliza el texto.
        Pasos sugeridos:
            - Convertir a minúsculas.
            - Eliminar URLs.
            - Eliminar caracteres especiales.
            - Eliminar espacios redundantes.
        Devuelve:
        El texto limpio.
        """
        if pd.isna(text):
            return ""
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"https?:\/\/\S+", "", text)
        text = re.sub(
            r"[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\u2600-\u27BF]+",
            "",
            text,
        )
        # Se conserva '#' para poder detectar hashtags tras la limpieza.
        text = re.sub(r"[^a-z0-9\s#]", "", text)
        return text

    def extract_hashtags(self, text: str) -> list[str]:
        """
        Extrae y devuelve una lista de hashtags presentes en el texto.
        Implementación sugerida:
            - Utilizar expresiones regulares para encontrar palabras que comiencen con '#' .
        """
        return re.findall(r"#(\w+)", text.lower())

    def analytics_hashtags_extended(self) -> dict[str, DataFrame]:
        """
        Realiza un análisis avanzado de hashtags sobre el conjunto de datos cargado (self.data).
        El método realiza los siguientes pasos:
            1. Aplica la función clean_text a la columna 'text' para normalizar los datos.
            2. Extrae los hashtags de cada texto usando extract_hashtags y los almacena en una nueva columna.
            3. Convierte la columna 'date' a tipo datetime y extrae solo la fecha (sin la hora).
            4. Explota la columna de hashtags para obtener una fila por cada hashtag, lo que facilita los cálculos de
            frecuencia
            5. Calcula tres análisis:
                - Frecuencia total de cada hashtag (overall).
                - Frecuencia de hashtags por usuario (by_user).
                - Evolución de la frecuencia de hashtags por día (by_date).

        Retorna un diccionario con tres DataFrames, con claves:
            'overall': DataFrame con columnas ['hashtag', 'frequency'].
            'by_user': DataFrame con columnas ['user_name', 'hashtag', 'frequency'].
            'by_date': DataFrame con columnas ['date', 'hashtag', 'frequency'].
        """
        if self.data is None:
            raise ValueError(
                "No hay datos: usa load_data() antes o llama a generate_hashtag_wordcloud()."
            )

        df = self.data.copy()
        df["text"] = df["text"].apply(self.clean_text)
        df["hashtags"] = df["text"].apply(self.extract_hashtags)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

        exploded_hashtags_df = df.explode("hashtags", ignore_index=True)
        exploded_hashtags_df = exploded_hashtags_df[
            exploded_hashtags_df["hashtags"].notna()
        ]

        overall = (
            exploded_hashtags_df.groupby("hashtags", as_index=False)
            .size()
            .rename(columns={"hashtags": "hashtag", "size": "frequency"})
            .sort_values("frequency", ascending=False)
            .reset_index(drop=True)
        )

        by_user = (
            exploded_hashtags_df.groupby(["user_name", "hashtags"], as_index=False)
            .size()
            .rename(columns={"hashtags": "hashtag", "size": "frequency"})
            .sort_values("frequency", ascending=False)
            .reset_index(drop=True)
        )

        by_date = (
            exploded_hashtags_df.groupby(["date", "hashtags"], as_index=False)
            .size()
            .rename(columns={"hashtags": "hashtag", "size": "frequency"})
            .sort_values(["date", "frequency"], ascending=[True, False])
            .reset_index(drop=True)
        )

        return {"overall": overall, "by_user": by_user, "by_date": by_date}

    def generate_hashtag_wordcloud(
        self,
        overall_df: DataFrame | None = None,
        max_words: int = DEFAULT_WORDCLOUD_MAX_WORDS,
        figsize: tuple[float, float] = (10, 6),
    ) -> None:
        """
        Genera y muestra una wordcloud basada en el análisis global de hashtags.
        Este método utiliza el DataFrame 'overall' que contiene la frecuencia global de cada hashtag.
        Si no se proporciona el DataFrame, se calcula llamando a analytics_hashtags_extended().
        Parámetros:
            - overall_df (pd.DataFrame, opcional): DataFrame con columnas ['hashtags', 'frequency']. Si es None, se
            calcula.
            - max_words (int, opcional): Número máximo de palabras a incluir en la wordcloud.
            - figsize (tuple, opcional): Tamaño de la figura a mostrar.
        Proceso:
            1. Si overall_df es None, llamar a analytics_hashtags_extended y extraer la parte 'overall'.
            2. Convertir el DataFrame a un diccionario donde las claves sean los hashtags y los valores sean las
            frecuencias.
            3. Utilizar la clase WordCloud de la librería wordcloud para generar la nube de palabras.
            4. Visualizar la wordcloud con matplotlib.
        """

        if self.data is None:
            self.load_data()

        if overall_df is None:
            overall_df = self.analytics_hashtags_extended()["overall"]

        if overall_df.empty:
            return

        freq_map = overall_df.set_index("hashtag")["frequency"].astype(float).to_dict()

        wc = WordCloud(
            width=800,
            height=400,
            max_words=max_words,
            background_color="white",
        )

        wc.generate_from_frequencies(freq_map)

        plt.figure(figsize=figsize)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()
