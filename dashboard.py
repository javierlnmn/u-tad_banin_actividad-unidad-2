from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

from data_loaders.kaggle import KaggleLoader
from main import DataExtractor

DATASET = "kaushiksuresh147/bitcoin-tweets"
FILE_PATH = "Bitcoin_tweets_dataset_2.csv"


@st.cache_data(show_spinner="Cargando CSV y analizando hashtags…")
def load_analytics(max_rows: int) -> tuple[dict[str, pd.DataFrame], int]:
    """
    max_rows <= 0 significa usar todas las filas (puede ser lento y pesado en memoria).
    """
    loader = KaggleLoader(dataset_name=DATASET, file_path=FILE_PATH)
    df = loader.load()
    total = len(df)
    if max_rows > 0:
        df = df.head(max_rows)
    extractor = DataExtractor(loader, data=df)
    return extractor.analytics_hashtags_extended(), total


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
        page_title="Hashtags — Bitcoin tweets",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Dashboard de hashtags")
    st.caption("Datos: Kaggle · `kaushiksuresh147/bitcoin-tweets`")

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
        if st.button("Limpiar caché y recargar"):
            st.cache_data.clear()
            st.rerun()

    stats, total_rows = load_analytics(int(max_rows))
    overall: pd.DataFrame = stats["overall"]
    by_user: pd.DataFrame = stats["by_user"]
    by_date: pd.DataFrame = stats["by_date"]
    used = int(max_rows) if int(max_rows) > 0 else total_rows

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas en el CSV", f"{total_rows:,}")
    c2.metric("Filas analizadas", f"{used:,}")
    c3.metric("Hashtags distintos", f"{len(overall):,}")
    c4.metric("Apariciones (suma freq.)", f"{int(overall['frequency'].sum()):,}")

    tab1, tab2, tab3, tab4 = st.tabs(["Ranking global", "Por usuario", "Por fecha", "Wordcloud"])

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
                mask = bu["hashtag"].astype(str).str.contains(hashtag_filter, case=False, na=False)
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
                bd_f = bd_f[bd_f["hashtag"].astype(str).str.contains(hashtag_filter, case=False, na=False)]
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
                fig_line.update_layout(height=480, legend=dict(orientation="h", yanchor="bottom", y=-0.4))
                st.plotly_chart(fig_line, use_container_width=True)
            st.dataframe(
                bd.sort_values(["date", "frequency"], ascending=[True, False]).head(300),
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


if __name__ == "__main__":
    main()
