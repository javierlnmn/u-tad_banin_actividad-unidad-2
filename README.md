# Entrega final — Redes sociales (NetworkX) + LLM local

Esta entrega amplía el proyecto con **minería de textos** y, además, **análisis de redes** e **interpretación con un LLM local** sobre tweets obtenidos por **API (RapidAPI, Twitter/X)**, centrados en **opiniones sobre la película de Michael Jackson** (*Michael*, biopic).

La primera entrega (Bitcoin / Kaggle, hashtags y wordcloud) sigue documentada en [`README_ENTREGA_1.md`](README_ENTREGA_1.md).


## Fuente de datos

* **API:** [RapidAPI](https://rapidapi.com) — host por defecto `twitter-api45.p.rapidapi.com` (ver `loaders/rapidapi_twitter.py`).
* **Consulta por defecto:** `michael jackson film opinions` (configurable en `DEFAULT_QUERY` del loader).
* **Caché local:** los tweets se guardan en **`data/rapidapi_tweets.csv`** tras una descarga correcta (columnas normalizadas: `username`, `text`, `date`).
* **Credenciales:** variable de entorno **`RAPIDAPI_KEY`** en `.env` cuando se llama a la API (no hace falta si solo se usa un CSV ya generado con `--use-file`).

El corpus mezcla **inglés y español**; el extractor puede aplicar **stopwords en ambos idiomas** (NLTK) al construir `clean_text` y al resumen.


## Metodología (resumen)

Todo pasa por `DataExtractor` (`extractor.py`):

| Paso | Qué hace |
|------|----------|
| **Limpieza** | `clean_text` + eliminación de stopwords **inglés/español** (NLTK); se mantienen tildes y `#`. |
| **`model_topics()`** | **LDA** (Gensim) sobre `clean_text` → tópicos como listas de palabras más probables (sin etiquetas automáticas). |
| **`analyze_sentiment()`** | **TextBlob**: polaridad y subjetividad (más fiable en inglés; aproximación en corpus mixto). |
| **`parse_and_summarize()`** | Resumen **extractivo**: se puntúan oraciones por frecuencia de términos (sin stopwords), se eligen las mejores con un ratio y un **tope de oraciones**. |
| **`build_interaction_graph()`** | Grafo **dirigido** de menciones `@usuario` (autor → mencionado, peso = frecuencia). |
| **`analyze_network()`** | Grado medio, densidad, **PageRank**, comunidades **Louvain** y visualización del grafo. |
| **`generate_prompt_from_network()`** | Prompt en español con top 3 centralidad y hashtag más frecuente para el LLM. |
| **`chat_local_llm()`** | Chat local con **`google/gemma-4-E2B-it`** (Hugging Face + PyTorch). |

En el **dashboard** se añaden **árboles de dependencia** (spaCy) solo como apoyo visual del análisis sintáctico, no como salida del extractor.


## Análisis de red + LLM (Unidad 2)

### Flujo

1. Cargar corpus (`rapidapi_tweets.csv` o API).
2. `build_interaction_graph()` — nodos = usuarios, aristas = menciones en el texto original.
3. `analyze_network(G)` — métricas y comunidades; imprime el **top 3 por PageRank**.
4. `generate_prompt_from_network(G)` — integra métricas y el hashtag más frecuente (plantilla en `llm/prompts/network_analysis.txt`).
5. `chat_local_llm(prompt=...)` — el modelo interpreta la red; luego modo chat interactivo.

### Ejecución por CLI

```bash
# Solo red + prompt (sin cargar el LLM)
python main.py --loader rapidapi --use-file --network --no-network-plot

# Red + chat con Gemma (GPU recomendada; aceptar licencia en Hugging Face)
python main.py --loader rapidapi --use-file --network --llm
```

| Opción | Descripción |
|--------|-------------|
| `--network` | Grafo, métricas y prompt. |
| `--llm` | Tras la red, carga el modelo y abre el chat. |
| `--llm-model ID` | Otro checkpoint de Hugging Face. |
| `--no-network-plot` | Omite la ventana matplotlib del grafo. |

Sin `--network` ni `--llm`, `main.py` sigue ejecutando LDA, sentimiento y resumen (entrega 2).

### Justificación pedagógica

- **Insights de red en el prompt:** el LLM no ve el CSV completo; recibe resúmenes estructurados (hubs, hashtag dominante, densidad). Así la interpretación se ancla en **evidencia cuantitativa** y se reduce la alucinación.
- **PageRank en grafos dirigidos:** prioriza nodos mencionados o que mencionan de forma relevante en la subred de interacciones, más adecuado que el grado simple en Twitter.
- **Louvain:** agrupa usuarios con interacción densa; ayuda a hipotetizar subcomunidades (críticos, fans, medios).
- **Ajuste fino (contexto del curso):** en *fine-tuning*, los **steps** son pasos de optimización sobre el corpus de entrenamiento; el **training loss** mide el error del modelo al predecir el siguiente token. Si el loss baja de forma estable, el modelo está adaptándose; si sube o oscila, conviene revisar learning rate o sobreajuste. En esta entrega **no** se hace fine-tuning: se usa un modelo **preentrenado** en inferencia local, lo que es más ligero para un entregable académico.


## Entorno e instalación

1. **Clona o coloca el proyecto** y entra en la carpeta raíz del repo.

2. **Crea y activa un entorno virtual** (recomendado):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   En **Windows** (PowerShell o CMD):

   ```text
   .venv\Scripts\activate
   ```

3. **Instala dependencias**:

   ```bash
   pip install -r requirements.txt
   ```

   Para el LLM local hace falta **PyTorch** y **transformers** (incluidos en `requirements.txt`). El modelo `google/gemma-4-E2B-it` requiere aceptar la licencia en [Hugging Face](https://huggingface.co/google/gemma-4-E2B-it) y, en la práctica, **GPU con suficiente VRAM** (~4–8 GB según cuantización).

4. **Modelo spaCy** (para los árboles de dependencias del dashboard):

   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **RapidAPI (solo si vas a descargar tweets tú mismo):** crea un fichero `.env` en la raíz con `RAPIDAPI_KEY=tu_clave`. Si solo usas un `data/rapidapi_tweets.csv` ya generado, no hace falta.

6. **NLTK:** el proyecto descarga automáticamente lo necesario (`stopwords`, `punkt_tab`, etc.) la primera vez que corre el extractor o el dashboard.


## Ejecución

### 1. Actualizar el corpus (`main.py`) — **opcional**

Solo hace falta si quieres **volver a descargar** tweets o aún **no tienes** `data/rapidapi_tweets.csv`. Si el fichero ya está en el repo (entrega, compañero, copia local), puedes **saltarte este paso** e ir directo al dashboard.

Descarga por la API y guarda en `data/rapidapi_tweets.csv` (o valida el CSV con `--use-file` sin gastar cuota):

```bash
python main.py --loader rapidapi
```

| Opción | Descripción |
|--------|-------------|
| `--rapidapi-tweet-count N` | Número de tweets (por defecto 300). |
| `--use-file` | Usa `data/rapidapi_tweets.csv` sin llamar a la API. |

```bash
python main.py --loader rapidapi --use-file
```

### 2. Dashboard Streamlit (`dashboard_rapidapi_tweets.py`)

Con `data/rapidapi_tweets.csv` presente, lanza la app (tópicos LDA, sentimiento, resúmenes, árboles spaCy). El dashboard **solo lee el fichero**, no la API; si cambias datos u opciones, usa **«Limpiar caché»** en la barra lateral.

```bash
streamlit run dashboard_rapidapi_tweets.py
```

El corpus Bitcoin / Kaggle de la entrega 1 se visualiza con `dashboard_kaggle_tweets.py` (ver [`README_ENTREGA_1.md`](README_ENTREGA_1.md)).


## Estructura relevante

| Ruta | Rol |
|------|-----|
| `loaders/rapidapi_twitter.py` | Cliente RapidAPI, query por defecto sobre la película MJ, guardado en CSV. |
| `extractor.py` | LDA, sentimiento, resumen, **grafo NetworkX**, **prompt** y **chat LLM**. |
| `llm/prompts/network_analysis.txt` | Plantilla del prompt para el análisis interpretativo de la red. |
| `llm/network_prompt.py` | Rellena la plantilla con métricas y hashtags. |
| `dashboard_rapidapi_tweets.py` | Visualización entrega 2. |
| `data/rapidapi_tweets.csv` | Corpus cacheado de la API. |
