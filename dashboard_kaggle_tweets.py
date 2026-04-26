from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

from extractor import DataExtractor
from loaders.kaggle import KaggleLoader

DATASET = "kaushiksuresh147/bitcoin-tweets"
FILE_PATH = "Bitcoin_tweets_dataset_2.csv"


@st.cache_data(show_spinner="Descargando / leyendo CSV…")
def load_df(max_rows: int) -> tuple[pd.DataFrame, int]:
    loader = KaggleLoader(dataset_name=DATASET, file_path=FILE_PATH)
    df = loader.load()
    total = len(df)
    if max_rows > 0:
        df = df.head(max_rows)
    return df.reset_index(drop=True), total


@st.cache_data(show_spinner="Analizando hashtags…")
def cached_hashtag_stats(max_rows: int) -> dict[str, pd.DataFrame]:
    df, _ = load_df(max_rows)
    extractor = DataExtractor(
        KaggleLoader(dataset_name=DATASET, file_path=FILE_PATH), data=df
    )
    return extractor.analytics_hashtags_extended()


@st.cache_data(show_spinner="LDA, sentimiento y resumen extractivo…")
def cached_nlp_pipeline(
    max_rows: int,
    num_topics: int,
    lda_passes: int,
    summary_ratio: float,
) -> tuple[list[list[str]], pd.DataFrame, str]:
    df, _ = load_df(max_rows)
    extractor = DataExtractor(
        KaggleLoader(dataset_name=DATASET, file_path=FILE_PATH), data=df.copy()
    )
    topics = extractor.model_topics(num_topics=num_topics, passes=lda_passes)
    sentiment_df = extractor.analyze_sentiment(method="textblob")
    summary = extractor.parse_and_summarize(summary_ratio=summary_ratio)
    return topics, sentiment_df, summary


def build_wordcloud_figure(overall: pd.DataFrame, max_words: int) -> plt.Figure:
    freq = overall.set_index("hashtag")["frequency"].astype(float).to_dict()
    wc = WordCloud(
        width=900,
        height=450,
        max_words=max_words,
        background_color="white",
    ).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Bitstream — Bitcoin (Kaggle)",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Bitstream")
    st.caption("Corpus Kaggle · `kaushiksuresh147/bitcoin-tweets` — hashtags y NLP")

    with st.sidebar:
        st.header("Opciones")
        max_rows = st.number_input(
            "Máximo de filas a procesar (0 = todas)",
            min_value=0,
            value=15_000,
            step=1_000,
            help="Limitar filas acelera la carga y el análisis en máquinas modestas.",
        )
        top_n = st.slider("Top hashtags en gráficos", 5, 50, 20)
        wc_words = st.slider("Palabras en la nube", 20, 200, 80)
        hashtag_filter = st.text_input(
            "Filtrar por hashtag (contiene)",
            "",
            help="Vacío = sin filtro. Ej.: bitcoin",
        ).strip()

        st.subheader("NLP (LDA / sentimiento / resumen)")
        num_topics = st.slider("Número de tópicos (LDA)", 2, 15, 5)
        lda_passes = st.slider("Pases LDA (más = más lento)", 1, 20, 5)
        summary_ratio = st.slider("Ratio resumen extractivo", 0.05, 0.9, 0.3, 0.05)

        if st.button("Limpiar caché y recargar"):
            st.cache_data.clear()
            st.rerun()

    max_r = int(max_rows)
    _, total_rows = load_df(max_r)
    stats = cached_hashtag_stats(max_r)
    topics, sentiment_df, summary = cached_nlp_pipeline(
        max_r, num_topics, lda_passes, summary_ratio
    )

    overall: pd.DataFrame = stats["overall"]
    by_user: pd.DataFrame = stats["by_user"]
    by_date: pd.DataFrame = stats["by_date"]
    used = max_r if max_r > 0 else total_rows

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas en el CSV", f"{total_rows:,}")
    c2.metric("Filas analizadas", f"{used:,}")
    c3.metric("Hashtags distintos", f"{len(overall):,}")
    c4.metric("Apariciones (suma freq.)", f"{int(overall['frequency'].sum()):,}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Ranking global",
            "Por usuario",
            "Por fecha",
            "Wordcloud",
            "Tópicos, sentimiento y resumen",
        ]
    )

    top = overall.head(top_n).copy()
    with tab1:
        st.subheader("Frecuencia global")
        fig_bar = px.bar(
            top,
            x="frequency",
            y="hashtag",
            orientation="h",
            labels={"frequency": "Frecuencia", "hashtag": "Hashtag"},
        )
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"}, height=520)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.dataframe(
            overall,
            use_container_width=True,
            hide_index=True,
            height=320,
        )

    with tab2:
        st.subheader("Hashtags por usuario")
        if by_user.empty:
            st.info("No hay columna `user_name` o no hay datos agregados.")
        else:
            bu = by_user
            if hashtag_filter:
                mask = (
                    bu["hashtag"]
                    .astype(str)
                    .str.contains(hashtag_filter, case=False, na=False)
                )
                bu = bu[mask]
            bu_show = bu.head(500)
            fig_u = px.scatter(
                bu_show.head(200),
                x="hashtag",
                y="user_name",
                size="frequency",
                size_max=40,
                hover_data=["frequency"],
                labels={"hashtag": "Hashtag", "user_name": "Usuario"},
            )
            fig_u.update_layout(height=560, xaxis_tickangle=-45)
            st.plotly_chart(fig_u, use_container_width=True)
            st.dataframe(bu_show, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Evolución por día (hashtags más frecuentes en el subset)")
        bd = by_date.dropna(subset=["date"])
        if bd.empty:
            st.warning("No hay datos temporales válidos en `date`.")
        else:
            top_tags = set(overall.head(min(8, len(overall)))["hashtag"].astype(str))
            bd_f = bd[bd["hashtag"].astype(str).isin(top_tags)]
            if hashtag_filter:
                bd_f = bd_f[
                    bd_f["hashtag"]
                    .astype(str)
                    .str.contains(hashtag_filter, case=False, na=False)
                ]
            if bd_f.empty:
                st.info("Sin puntos tras el filtro; prueba otro hashtag o más filas.")
            else:
                fig_line = px.line(
                    bd_f.sort_values("date"),
                    x="date",
                    y="frequency",
                    color="hashtag",
                    markers=False,
                    labels={"date": "Fecha", "frequency": "Frecuencia"},
                )
                fig_line.update_layout(
                    height=480, legend=dict(orientation="h", yanchor="bottom", y=-0.4)
                )
                st.plotly_chart(fig_line, use_container_width=True)
            st.dataframe(
                bd.sort_values(["date", "frequency"], ascending=[True, False]).head(
                    300
                ),
                use_container_width=True,
                hide_index=True,
            )

    with tab4:
        st.subheader("Nube de palabras (hashtags)")
        if overall.empty:
            st.warning("No hay hashtags para mostrar.")
        else:
            fig_wc = build_wordcloud_figure(overall, wc_words)
            st.pyplot(fig_wc, clear_figure=True)
            plt.close(fig_wc)

    with tab5:
        st.subheader("Tópicos (model_topics · LDA)")
        topics_table = pd.DataFrame(
            {
                "tópico": [f"#{i}" for i in range(len(topics))],
                "palabras": [", ".join(words) for words in topics],
            }
        )
        st.dataframe(topics_table, use_container_width=True, hide_index=True)
        for i, words in enumerate(topics):
            st.markdown(f"**Tópico {i}:** {', '.join(words)}")

        st.subheader("Sentimiento (analyze_sentiment · TextBlob)")
        if sentiment_df.empty:
            st.warning("No hay filas para analizar el sentimiento.")
        else:
            desc = sentiment_df[
                ["sentiment_polarity", "sentiment_subjectivity"]
            ].describe()
            st.dataframe(desc, use_container_width=True)

            col_a, col_b = st.columns(2)
            sample = sentiment_df
            if len(sentiment_df) > 2000:
                sample = sentiment_df.sample(2000, random_state=0)
            with col_a:
                fig_pol = px.histogram(
                    sentiment_df,
                    x="sentiment_polarity",
                    nbins=40,
                    labels={"sentiment_polarity": "Polaridad"},
                    title="Distribución de polaridad",
                )
                st.plotly_chart(fig_pol, use_container_width=True)
            with col_b:
                fig_sc = px.scatter(
                    sample,
                    x="sentiment_polarity",
                    y="sentiment_subjectivity",
                    opacity=0.35,
                    labels={
                        "sentiment_polarity": "Polaridad",
                        "sentiment_subjectivity": "Subjetividad",
                    },
                    title="Polaridad vs subjetividad (muestra)",
                )
                st.plotly_chart(fig_sc, use_container_width=True)

            show_cols = [
                c
                for c in (
                    "text",
                    "clean_text",
                    "sentiment_polarity",
                    "sentiment_subjectivity",
                )
                if c in sentiment_df.columns
            ]
            st.dataframe(
                sentiment_df[show_cols].head(500),
                use_container_width=True,
                hide_index=True,
                height=360,
            )

        st.subheader("Resumen extractivo (parse_and_summarize)")
        st.markdown(summary if summary else "_Sin texto de resumen._")


if __name__ == "__main__":
    main()
