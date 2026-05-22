from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Literal

import matplotlib.pyplot as plt
import networkx as nx
import nltk
import pandas as pd
import spacy
import torch
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from networkx.algorithms import community as nx_community
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from pandas import DataFrame
from textblob import TextBlob
from wordcloud import WordCloud

from llm.network_prompt import build_network_prompt
from loaders.base import DataLoader

DEFAULT_WORDCLOUD_MAX_WORDS = 100
DEFAULT_LLM_MODEL = "google/gemma-4-E2B-it"
MENTION_RE = re.compile(r"@([A-Za-z0-9_]+)")

nltk.download("stopwords")
nltk.download("punkt_tab")


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
        self._llm_tokenizer = None
        self._llm_model = None
        self._chat_history: list[dict[str, str]] = []
        self._last_network_metrics: dict[str, Any] | None = None

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
            - Utilizar expresiones regulares para encontrar palabras que comiencen con
            '#' .
        """
        return re.findall(r"#(\w+)", text.lower())

    def analytics_hashtags_extended(self) -> dict[str, DataFrame]:
        """
        Realiza un análisis avanzado de hashtags sobre el conjunto de datos cargado
        (self.data).
        El método realiza los siguientes pasos:
            1. Aplica la función clean_text a la columna 'text' para normalizar los
            datos.
            2. Extrae los hashtags de cada texto usando extract_hashtags y los almacena
            en una nueva columna.
            3. Convierte la columna 'date' a tipo datetime y extrae solo la fecha
            (sin la hora).
            4. Explota la columna de hashtags para obtener una fila por cada hashtag,
            lo que facilita los cálculos de
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
                "No hay datos: usa load_data() o agenerate_hashtag_wordcloud() antes."
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
        Este método utiliza el DataFrame 'overall' que contiene la frecuencia global de
        cada hashtag.
        Si no se proporciona el DataFrame, se calcula llamando a
        analytics_hashtags_extended().
        Parámetros:
            - overall_df (pd.DataFrame, opcional): DataFrame con columnas
            ['hashtags', 'frequency']. Si es None, se
            calcula.
            - max_words (int, opcional): Número máximo de palabras a incluir en la
            wordcloud.
            - figsize (tuple, opcional): Tamaño de la figura a mostrar.
        Proceso:
            1. Si overall_df es None, llamar a analytics_hashtags_extended y extraer la
            parte 'overall'.
            2. Convertir el DataFrame a un diccionario donde las claves sean los
            hashtags y los valores sean las
            frecuencias.
            3. Utilizar la clase WordCloud de la librería wordcloud para generar la
            nube de palabras.
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

    def _remove_stopwords(self, text: str) -> str:
        """
        Elimina las stopwords de un texto.
        """
        stopwords_list = set(stopwords.words("english"))
        return " ".join([word for word in text.split() if word not in stopwords_list])

    def _check_clean_text(self) -> None:
        """
        Asegura que la columna 'clean_text' existe.
        """
        if self.data is None:
            self.load_data()

        if "clean_text" not in self.data.columns:
            self.data["clean_text"] = self.data["text"].apply(self.clean_text)
            self.data["clean_text"] = self.data["clean_text"].apply(
                self._remove_stopwords
            )

    def model_topics(
        self, num_topics: int = 5, passes: int = 10, show_visualization: bool = True
    ) -> list[list[str]]:
        """
        Aplica el modelo LDA para descubrir tópicos en el corpus.
        Pasos:
        1. Asegurarse de que la columna 'clean_text' existe (se debe llamar previamente
        a clean_text).
        2. Tokeniza la columna 'clean_text' (división simple en palabras).
        3. Crea un diccionario y un corpus (bag-of-words) a partir de los tokens.
        4. Entrena el modelo LDA con los parámetros especificados.
        5. Extrae y muestra los tópicos en formato lista (cada tópico es una lista de
        palabras).
        Devuelve:
        Lista de tópicos, por ejemplo: [['word1', 'word2', ...], ['word3', ...], ...]
        """
        self._check_clean_text()

        df = self.data.copy()

        df["tokens"] = df["clean_text"].apply(simple_preprocess, deacc=True)

        dictionary = corpora.Dictionary(df["tokens"])
        corpus = [dictionary.doc2bow(text) for text in df["tokens"]]

        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
        )

        return [
            [word for word, _ in topic]
            for idx, topic in lda_model.show_topics(
                num_topics=num_topics, num_words=10, formatted=False
            )
        ]

    def analyze_sentiment(
        self, method: Literal["textblob", "spacy"] = "textblob"
    ) -> pd.DataFrame:
        """
        Analiza el sentimiento de cada tweet utilizando el método especificado.
        Parámetros:
        - method: 'textblob' o 'spacy'. Si se elige 'spacy', se usará spacytextblob.
        Proceso:
        - Para cada 'clean_text', calcula la polaridad y subjetividad.
        - Almacena los resultados en las columnas 'sentiment_polarity' y
        'sentiment_subjectivity'.
        Devuelve:
        DataFrame actualizado con las nuevas columnas de sentimiento.
        """
        self._check_clean_text()

        df = self.data.copy()

        if method == "spacy":
            out = self._analyze_sentiment_spacy(df)
        elif method == "textblob":
            out = self._analyze_sentiment_textblob(df)
        else:
            raise ValueError(
                f"Método de análisis de sentimiento desconocido: {method!r}"
            )

        self.data = out
        return out

    def _analyze_sentiment_textblob(self, df: pd.DataFrame) -> pd.DataFrame:
        df["sentiment_polarity"] = df["clean_text"].apply(
            lambda t: float("nan") if t == "" else float(TextBlob(t).sentiment.polarity)
        )
        df["sentiment_subjectivity"] = df["clean_text"].apply(
            lambda t: (
                float("nan") if t == "" else float(TextBlob(t).sentiment.subjectivity)
            )
        )
        return df

    def _analyze_sentiment_spacy(self, df: pd.DataFrame) -> pd.DataFrame:
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("spacytextblob")

        doc = nlp(str(df["clean_text"]))
        df["sentiment_polarity"] = df["clean_text"].apply(
            lambda t: float("nan") if t == "" else float(doc._.blob.polarity)
        )
        df["sentiment_subjectivity"] = df["clean_text"].apply(
            lambda t: float("nan") if t == "" else float(doc._.blob.subjectivity)
        )
        df["sentiment_tokens_pos"] = df["clean_text"].apply(
            lambda t: {} if t == "" else {tok.text: tok.pos_ for tok in doc}
        )
        return df

    def parse_and_summarize(
        self,
        summary_ratio: float = 0.3,
        max_sentences: int | None = None,
    ) -> str:
        """
        Realiza un análisis de parsing y genera un resumen extractivo del corpus.
        Pasos:
        1. Concatena todos los textos limpios.
        2. Divide el texto concatenado en oraciones.
        3. Calcula una puntuación para cada oración basándose en la frecuencia de
        palabras (excluyendo stopwords).
        4. Selecciona las oraciones con mayor puntuación según el ratio especificado.
        5. Devuelve el resumen formado por las oraciones seleccionadas, manteniendo el
        orden original.
        Parámetros:
        - summary_ratio: Proporción de oraciones a retener (ej. 0.3 para el 30%).
        - max_sentences: Si no es None, no se incluyen más de tantas oraciones aunque
          el ratio pida más (útil cuando el corpus tiene miles de oraciones).
        Devuelve:
        Un string con el resumen generado.
        """
        self._check_clean_text()

        concat_texts = "\n".join(self.data["clean_text"].tolist())
        original_sentences = [
            s.strip() for s in sent_tokenize(concat_texts) if s.strip()
        ]

        word_counts = {}

        for sent in original_sentences:
            for w in word_tokenize(sent):
                word_counts[w] = word_counts.get(w, 0) + 1

        max_freq = max(word_counts.values()) if word_counts else 1

        sentence_scores = {}
        for idx, sent in enumerate(original_sentences):
            tokens = word_tokenize(sent)
            score = 0
            for w in tokens:
                if w in word_counts:
                    score += word_counts[w] / max_freq
            sentence_scores[idx] = score

        sorted_sentences = sorted(
            sentence_scores.items(), key=lambda x: x[1], reverse=True
        )

        total_sentences = len(original_sentences)
        top_n = math.ceil(total_sentences * summary_ratio)
        if max_sentences is not None:
            top_n = min(top_n, max_sentences)
        top_n = min(top_n, total_sentences)
        if top_n <= 0:
            return ""

        selected_idx = [idx for idx, score in sorted_sentences[:top_n]]
        selected_idx.sort()
        summary_sentences = [original_sentences[i] for i in selected_idx]

        summary = " ".join(summary_sentences)
        return summary

    @staticmethod
    def _extract_mentions(text: object) -> list[str]:
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return []
        return [m.lower() for m in MENTION_RE.findall(str(text))]

    def build_interaction_graph(self) -> nx.DiGraph:
        """
        Construye un grafo de interacciones a partir de los datos.
        Se asume que self.data tiene las columnas 'user_name' y 'text' para extraer menciones.
        """
        if self.data is None:
            self.load_data()

        graph = nx.DiGraph()

        for _, row in self.data.iterrows():
            author = str(row["username"]).strip().lower()
            if not author:
                continue
            graph.add_node(author)
            for mention in self._extract_mentions(row["text"]):
                if not mention or mention == author:
                    continue
                graph.add_node(mention)
                if graph.has_edge(author, mention):
                    graph[author][mention]["weight"] += 1
                else:
                    graph.add_edge(author, mention, weight=1)

        return graph

    def _compute_network_metrics(self, graph: nx.DiGraph) -> dict[str, Any]:
        """Calcula métricas y comunidades (Louvain) sin imprimir ni graficar."""
        nodes = graph.number_of_nodes()
        edges = graph.number_of_edges()

        if nodes == 0:
            return {
                "nodes": 0,
                "edges": 0,
                "avg_in_degree": 0.0,
                "avg_out_degree": 0.0,
                "density": 0.0,
                "top_centrality": [],
                "communities": [],
                "pagerank": {},
                "in_degree": {},
                "out_degree": {},
            }

        in_deg = dict(graph.in_degree())
        out_deg = dict(graph.out_degree())
        avg_in = sum(in_deg.values()) / nodes
        avg_out = sum(out_deg.values()) / nodes

        try:
            density = float(nx.density(graph))
        except nx.NetworkXError:
            density = 0.0

        pagerank = nx.pagerank(graph, weight="weight")
        top_centrality = sorted(
            pagerank.items(), key=lambda item: item[1], reverse=True
        )[:3]

        undirected = graph.to_undirected()
        if edges > 0:
            communities = list(
                nx_community.louvain_communities(undirected, weight="weight", seed=42)
            )
        else:
            communities = [{n} for n in graph.nodes()]

        return {
            "nodes": nodes,
            "edges": edges,
            "avg_in_degree": avg_in,
            "avg_out_degree": avg_out,
            "density": density,
            "top_centrality": top_centrality,
            "communities": communities,
            "pagerank": pagerank,
            "in_degree": in_deg,
            "out_degree": out_deg,
        }

    def analyze_network(
        self,
        graph: nx.DiGraph,
        *,
        show_plot: bool = True,
        figsize: tuple[float, float] = (10, 8),
    ) -> dict[str, Any]:
        """
        Calcula métricas de red y detecta comunidades utilizando el algoritmo de Louvain.
        Imprime estadísticas y genera una visualización básica.
        """
        metrics = self._compute_network_metrics(graph)
        self._last_network_metrics = metrics

        print("\n=== Análisis de red (NetworkX) ===")
        print(f"Nodos: {metrics['nodes']} | Aristas: {metrics['edges']}")
        print(
            f"Grado medio (entrada/salida): "
            f"{metrics['avg_in_degree']:.2f} / {metrics['avg_out_degree']:.2f}"
        )
        print(f"Densidad: {metrics['density']:.4f}")

        if metrics["top_centrality"]:
            print("\nTop 3 nodos por centralidad (PageRank):")
            for rank, (node, score) in enumerate(metrics["top_centrality"], start=1):
                print(f"  {rank}. @{node} — {score:.4f}")

        print(f"\nComunidades detectadas (Louvain): {len(metrics['communities'])}")
        for idx, comm in enumerate(metrics["communities"], start=1):
            members = ", ".join(f"@{u}" for u in sorted(comm))
            print(f"  Comunidad {idx} ({len(comm)} nodos): {members}")

        if show_plot and metrics["nodes"] > 0:
            plt.figure(figsize=figsize)
            layout = nx.spring_layout(
                graph, seed=42, k=0.9 / math.sqrt(metrics["nodes"])
            )
            edge_weights = [graph[u][v].get("weight", 1) for u, v in graph.edges()]
            nx.draw_networkx_nodes(graph, layout, node_size=400, alpha=0.85)
            nx.draw_networkx_labels(graph, layout, font_size=8)
            nx.draw_networkx_edges(
                graph,
                layout,
                width=[0.5 + 0.3 * w for w in edge_weights],
                alpha=0.6,
                arrows=True,
                arrowsize=12,
            )
            plt.title("Grafo de interacciones por menciones (@usuario)")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close()

        return metrics

    def generate_prompt_from_network(self, graph: nx.DiGraph) -> str:
        """
        Genera un prompt para el LLM utilizando insights del análisis de la red.
        Se extraen métricas como el top 3 de nodos por centralidad y el hashtag más frecuente.
        La función construye un prompt que pide explicar posibles razones de las tendencias observadas
        """
        metrics = self._last_network_metrics
        if metrics is None:
            metrics = self._compute_network_metrics(graph)
            self._last_network_metrics = metrics

        if self.data is None:
            self.load_data()
        all_hashtags: list[str] = []
        for text in self.data["text"]:
            all_hashtags.extend(self.extract_hashtags(self.clean_text(text)))
        hashtag_counts = Counter(all_hashtags)
        if hashtag_counts:
            top_hashtag, top_hashtag_freq = hashtag_counts.most_common(1)[0]
        else:
            top_hashtag, top_hashtag_freq = "ninguno detectado", 0

        return build_network_prompt(
            nodes=metrics["nodes"],
            edges=metrics["edges"],
            density=metrics["density"],
            communities_count=len(metrics["communities"]),
            top_centrality=metrics["top_centrality"],
            top_hashtag=top_hashtag,
            top_hashtag_freq=top_hashtag_freq,
        )

    def _load_local_llm(self, model_id: str = DEFAULT_LLM_MODEL) -> None:
        """Carga perezosa del tokenizer y del modelo causal en local."""
        if self._llm_model is not None and self._llm_tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        print(f"Cargando modelo local {model_id!r} en {device}…")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            model = model.to(device)

        model.eval()
        self._llm_tokenizer = tokenizer
        self._llm_model = model

    def _generate_llm_reply(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 512,
        model_id: str = DEFAULT_LLM_MODEL,
    ) -> str:

        self._load_local_llm(model_id)
        assert self._llm_tokenizer is not None
        assert self._llm_model is not None

        tokenizer = self._llm_tokenizer
        model = self._llm_model

        if hasattr(tokenizer, "apply_chat_template"):
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            input_ids = tokenizer(text, return_tensors="pt").input_ids

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        input_len = input_ids.shape[-1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0, input_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def chat_local_llm(
        self,
        prompt: str | None = None,
        *,
        model_id: str = DEFAULT_LLM_MODEL,
        max_new_tokens: int = 512,
    ) -> list[dict[str, str]]:
        """
        Levanta un modelo LLM preentrenado (gemma-4-e2b-it) y permite la interacción en modo chat.
        Si se proporciona un prompt (por ejemplo, generado a partir de la red), se utiliza como
        mensaje inicial para generar una respuesta automática, que se incorpora al contexto de la
        conversación.
        """
        self._chat_history = []

        if prompt:
            print("\n--- Prompt inicial (insights de la red) ---\n")
            print(prompt)
            self._chat_history.append({"role": "user", "content": prompt})
            reply = self._generate_llm_reply(
                self._chat_history,
                max_new_tokens=max_new_tokens,
                model_id=model_id,
            )
            print("\n--- Respuesta del modelo ---\n")
            print(reply)
            self._chat_history.append({"role": "assistant", "content": reply})

        print(
            "\nModo chat (escribe 'salir' para terminar). "
            "El contexto incluye el análisis previo si se proporcionó un prompt inicial."
        )

        while True:
            try:
                user_text = input("\nTú: ").strip()
            except KeyboardInterrupt:
                print("\nFin del chat.")
                break

            if user_text.lower() in {"salir", "exit", "quit"}:
                print("Fin del chat.")
                break
            if not user_text:
                continue

            self._chat_history.append({"role": "user", "content": user_text})
            reply = self._generate_llm_reply(
                self._chat_history,
                max_new_tokens=max_new_tokens,
                model_id=model_id,
            )
            print(f"\nAsistente: {reply}")
            self._chat_history.append({"role": "assistant", "content": reply})

        return self._chat_history
