from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from extractor import DataExtractor
from loaders.rapidapi_twitter import RapidAPITwitterLoader

load_dotenv()


def normalize_rapidapi_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "username" in out.columns and "user_name" not in out.columns:
        out = out.rename(columns={"username": "user_name"})
    return out


def polarity_label(p: float) -> str:
    if pd.isna(p):
        return "sin dato"
    if p > 0.1:
        return "positivo"
    if p < -0.1:
        return "negativo"
    return "neutral"


@st.cache_data(show_spinner="Leyendo data/rapidapi_tweets.csv…")
def load_corpus(tweet_count: int) -> tuple[pd.DataFrame, int]:
    loader = RapidAPITwitterLoader(tweet_count=tweet_count, use_file=True)
    df = normalize_rapidapi_columns(loader.load())
    return df.reset_index(drop=True), len(df)


@st.cache_data(show_spinner="LDA, sentimiento (TextBlob) y resúmenes…")
def run_extractor_pipeline(
    tweet_count: int,
    num_topics: int,
    lda_passes: int,
    summary_ratios: tuple[float, ...],
    max_summary_sentences: int,
) -> tuple[list[list[str]], pd.DataFrame, dict[float, str], int]:
    df, n = load_corpus(tweet_count)
    loader = RapidAPITwitterLoader(tweet_count=tweet_count, use_file=True)
    extractor = DataExtractor(loader=loader, data=df.copy())

    topics = extractor.model_topics(num_topics=num_topics, passes=lda_passes)
    sentiment_df = extractor.analyze_sentiment(method="textblob")

    summaries: dict[float, str] = {}
    for r in summary_ratios:
        summaries[float(r)] = extractor.parse_and_summarize(
            summary_ratio=float(r),
            max_sentences=max_summary_sentences,
        )

    return topics, sentiment_df, summaries, n


@st.cache_resource(show_spinner="Cargando modelo spaCy…")
def spacy_nlp():
    import spacy

    return spacy.load("en_core_web_sm")


def dependency_tree_html(text: str, nlp, max_chars: int = 400) -> str:
    from spacy import displacy

    snippet = (text or "").strip()
    if not snippet:
        return ""
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 1] + "…"
    doc = nlp(snippet)
    return displacy.render(
        doc,
        style="dep",
        jupyter=False,
        minify=True,
        options={"compact": True, "distance": 110},
    )


def main() -> None:
    st.set_page_config(
        page_title="RapidAPI — Tópicos, sentimiento y resúmenes",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Corpus RapidAPI: tópicos, sentimiento y resúmenes")
    st.caption(
        "Datos: `data/rapidapi_tweets.csv` · Métodos: `model_topics`, "
        "`analyze_sentiment`, `parse_and_summarize`."
    )

    with st.sidebar:
        st.header("Datos")
        st.info(
            "Solo lectura del fichero `data/rapidapi_tweets.csv`. "
            "Para refrescar tweets, ejecuta la API desde la terminal "
            "(p. ej. `python main.py --loader rapidapi`) y vuelve a abrir el dashboard."
        )
        tweet_count = st.number_input(
            "Número de tweets",
            min_value=1,
            max_value=5000,
            value=300,
            step=50,
        )
        st.header("Modelos")
        num_topics = st.slider("Tópicos LDA", 2, 12, 5)
        lda_passes = st.slider("Pases LDA", 1, 15, 5)
        st.caption(
            "Resúmenes: el ratio elige candidatas; "
            "el tope de oraciones acorta el texto."
        )
        max_sents = st.slider(
            "Máx. oraciones por resumen",
            min_value=3,
            max_value=40,
            value=15,
            help=(
                "Tope duro: evita resúmenes enormes si hay muchas frases en el corpus."
            ),
        )
        r1 = st.slider("Resumen A — ratio", 0.02, 0.35, 0.08, 0.01)
        r2 = st.slider("Resumen B — ratio", 0.02, 0.35, 0.12, 0.01)
        r3 = st.slider("Resumen C — ratio", 0.02, 0.35, 0.18, 0.01)
        st.header("Árboles de dependencia")
        n_trees = st.slider("Ejemplos a visualizar", 1, 6, 3)
        if st.button("Limpiar caché"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    ratios = tuple(sorted({float(r1), float(r2), float(r3)}))
    tc = int(tweet_count)

    try:
        topics, sentiment_df, summaries, n_loaded = run_extractor_pipeline(
            tc,
            num_topics,
            lda_passes,
            ratios,
            int(max_sents),
        )
    except Exception as err:
        st.error(f"No se pudieron obtener los datos o ejecutar el pipeline: {err}")
        st.stop()

    st.metric("Tweets en el análisis", f"{n_loaded:,}")

    # --- 1. Tópicos: tabla + interpretación ---
    st.header("1. Tópicos LDA (`model_topics`)")
    st.markdown(
        "Cada fila resume las palabras más asociadas a un tópico. "
        "LDA asume mezclas de tópicos por documento; la interpretación "
        "es orientativa hasta contrastarla con tweets concretos."
    )

    topic_rows = [
        {"tópico": i, "palabras_clave": ", ".join(words)}
        for i, words in enumerate(topics)
    ]
    st.dataframe(
        pd.DataFrame(topic_rows),
        use_container_width=True,
        hide_index=True,
    )

    # --- 2. Sentimiento: gráficos, tablas, árboles ---
    st.header("2. Sentimiento (`analyze_sentiment` con TextBlob)")
    st.markdown(
        "Distribución global y ejemplos tabulados. Los **árboles** son de "
        "**dependencias sintácticas** (spaCy `en_core_web_sm`), útiles para ver "
        "sujeto, verbo y complementos en frases ejemplo."
    )

    if sentiment_df.empty:
        st.warning("No hay filas de sentimiento.")
    else:
        pol = sentiment_df["sentiment_polarity"].dropna()
        sub = sentiment_df["sentiment_subjectivity"].dropna()

        c1, c2 = st.columns(2)
        with c1:
            fig_p = px.histogram(
                sentiment_df,
                x="sentiment_polarity",
                nbins=35,
                labels={"sentiment_polarity": "Polaridad (−1 … 1)"},
                title="Distribución de polaridad",
            )
            st.plotly_chart(fig_p, use_container_width=True)
        with c2:
            fig_s = px.histogram(
                sentiment_df,
                x="sentiment_subjectivity",
                nbins=35,
                labels={"sentiment_subjectivity": "Subjetividad (0 … 1)"},
                title="Distribución de subjetividad",
            )
            st.plotly_chart(fig_s, use_container_width=True)

        sentiment_df = sentiment_df.copy()
        sentiment_df["etiqueta"] = sentiment_df["sentiment_polarity"].apply(
            polarity_label
        )
        counts = (
            sentiment_df["etiqueta"]
            .value_counts()
            .reindex(
                ["positivo", "neutral", "negativo", "sin dato"],
                fill_value=0,
            )
        )
        fig_bar = px.bar(
            x=counts.index.astype(str),
            y=counts.values,
            labels={"x": "Sentimiento (heurística TextBlob)", "y": "Nº de tweets"},
            title="Conteo por categoría aproximada",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        if len(pol) > 1 and len(sub) > 1:
            scat = sentiment_df.dropna(
                subset=["sentiment_polarity", "sentiment_subjectivity"]
            )
            scat_kwargs: dict = {
                "x": "sentiment_polarity",
                "y": "sentiment_subjectivity",
                "color": "etiqueta",
                "labels": {
                    "sentiment_polarity": "Polaridad",
                    "sentiment_subjectivity": "Subjetividad",
                },
                "title": "Polaridad vs subjetividad",
                "opacity": 0.65,
            }
            if "text" in scat.columns:
                scat_kwargs["hover_data"] = ["text"]
            fig_sc = px.scatter(scat, **scat_kwargs)
            st.plotly_chart(fig_sc, use_container_width=True)

        st.subheader("Tabla de ejemplos (texto y puntuaciones)")
        show_cols = [
            c
            for c in (
                "text",
                "sentiment_polarity",
                "sentiment_subjectivity",
                "etiqueta",
            )
            if c in sentiment_df.columns
        ]
        st.dataframe(
            sentiment_df[show_cols].head(80),
            use_container_width=True,
            hide_index=True,
            height=360,
        )

        st.subheader("Ejemplos de árboles de dependencias (spaCy)")
        try:
            nlp = spacy_nlp()
        except Exception as e:
            st.warning(
                f"No se pudo cargar `en_core_web_sm` (árboles desactivados): {e}. "
                "Instala con: `python -m spacy download en_core_web_sm`"
            )
            nlp = None

        if nlp is not None:
            text_col = "text" if "text" in sentiment_df.columns else "clean_text"
            candidates = (
                sentiment_df[sentiment_df[text_col].astype(str).str.len() > 30]
                .dropna(subset=[text_col])
                .head(max(n_trees * 3, 12))
            )
            shown = 0
            for _, row in candidates.iterrows():
                if shown >= n_trees:
                    break
                t = str(row[text_col]).strip()
                if len(t) < 25:
                    continue
                lbl = polarity_label(row.get("sentiment_polarity"))
                st.markdown(f"**Tweet {shown + 1}** ({lbl})")
                st.caption(t[:280] + ("…" if len(t) > 280 else ""))
                html = dependency_tree_html(t, nlp)
                if html:
                    wrap = (
                        "<div style='overflow-x:auto;border:1px solid #eee;"
                        f"padding:8px'>{html}</div>"
                    )
                    components.html(wrap, height=420, scrolling=True)
                shown += 1
            if shown == 0:
                st.info("No hay textos suficientemente largos para dibujar árboles.")

    # --- 3. Resúmenes ---
    st.header("3. Resúmenes extractivos (`parse_and_summarize`)")
    st.markdown(
        "Mismo corpus unido: se puntúan oraciones por frecuencia de términos, "
        "se eligen las mejores según el ratio y **nunca más** que el tope de "
        "oraciones de la barra lateral."
    )

    sorted_ratios = sorted(summaries.keys())
    for r in sorted_ratios:
        text = summaries[r]
        title = f"Resumen — ratio ≈ {r:.2f}"
        with st.expander(title, expanded=(r == sorted_ratios[0])):
            st.write(text if text else "_(vacío)_")


if __name__ == "__main__":
    main()
