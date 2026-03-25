from data_loaders.base import DataLoader
from data_loaders.kaggle import KaggleLoader
import re
from pandas import DataFrame


class DataExtractor:
    def __init__(self, loader: DataLoader, chunksize: int = 10_000):
        """
        Inicializa el extractor con el archivo de origen.
        Parámetro:
        source_file: Ruta al archivo de datos (CSV o JSON).
        chunksize: Tamaño de los chunks para el procesamiento.
        loader: Cargador de datos para diferentes formatos.
        """
        self.loader: DataLoader = loader
        self.data: DataFrame = None
        self.chunksize = chunksize

    def process_data(self):
        """
        Método principal para procesar los datos.
            1. Carga los datos del archivo de origen.
            2. Extrae los hashtags del texto.
            3. Limpia el texto.
            4. Encuentra palabras clave en el texto.
        """
        self.data = self.loader.load()

        hashtags_from_text = self.data["text"].apply(self.extract_hashtags)
        # Sobreescribimos la columna hashtags con la lista de hashtags de cada texto
        self.data["hashtags"] = hashtags_from_text

        self.data["text"] = self.data["text"].apply(self.clean_text)

        keywords_from_text = self.data["text"].apply(self.extract_keywords)

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
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"https?:\/\/.*?[\s+]", "", text)
        text = re.sub(
            r"[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\u2600-\u27BF]+",
            "",
            text,
        )
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text

    def extract_hashtags(self, text: str) -> list:
        """
        Extrae y devuelve una lista de hashtags presentes en el texto.
        Implementación sugerida:
            - Utilizar expresiones regulares para encontrar palabras que comiencen con '#' .
        """
        hashtags = re.findall(r"#(\w+)", text)
        return hashtags

    # def analytics_hashtags_extended(self) -> dict:
    #     """
    #     Realiza un análisis avanzado de hashtags sobre el conjunto de datos cargado (self.data).
    #     El método realiza los siguientes pasos:
    #         1. Aplica la función clean_text a la columna 'text' para normalizar los datos.
    #         2. Extrae los hashtags de cada texto usando extract_hashtags y los almacena en una nueva columna.
    #         3. Convierte la columna 'date' a tipo datetime y extrae solo la fecha (sin la hora).
    #         4. Explota la columna de hashtags para obtener una fila por cada hashtag, lo que facilita los cálculos de
    #         frecuencia
    #         5. Calcula tres análisis:
    #             - Frecuencia total de cada hashtag (overall).
    #             - Frecuencia de hashtags por usuario (by_user).
    #             - Evolución de la frecuencia de hashtags por día (by_date).
    #     Retorna un diccionario con tres DataFrames, con claves:
    #         'overall': DataFrame con columnas ['hashtag', 'frequency'].
    #         'by_user': DataFrame con columnas ['user_name', 'hashtag', 'frequency'].
    #         'by_date': DataFrame con columnas ['date', 'hashtag', 'frequency'].
    #     """
    #     return {"overall": overall, "by_user": by_user, "by_date": by_date}

    # def generate_hashtag_wordcloud(
    #     self,
    #     overall_df: pd.DataFrame = None,
    #     max_words: int = 100,
    #     figsize: tuple = (10, 6),
    # ) -> None:
    #     """
    #     Genera y muestra una wordcloud basada en el análisis global de hashtags.
    #     Este método utiliza el DataFrame 'overall' que contiene la frecuencia global de cada hashtag.
    #     Si no se proporciona el DataFrame, se calcula llamando a analytics_hashtags_extended().
    #     Parámetros:
    #         overall_df (pd.DataFrame, opcional): DataFrame con columnas ['hashtags', 'frequency']. Si es None, se
    #         calcula.
    #         max_words (int, opcional): Número máximo de palabras a incluir en la wordcloud.
    #         figsize (tuple, opcional): Tamaño de la figura a mostrar.
    #     Proceso:
    #         1. Si overall_df es None, llamar a analytics_hashtags_extended y extraer la parte 'overall'.
    #         2. Convertir el DataFrame a un diccionario donde las claves sean los hashtags y los valores sean las
    #         frecuencias.
    #         3. Utilizar la clase WordCloud de la librería wordcloud para generar la nube de palabras.
    #         4. Visualizar la wordcloud con matplotlib.
    #     """


if __name__ == "__main__":
    loader = KaggleLoader(
        dataset_name="kaushiksuresh147/bitcoin-tweets",
        file_path="Bitcoin_tweets_dataset_2.csv",
    )

    extractor = DataExtractor(loader=loader)
    extractor.process_data()
