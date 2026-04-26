# Entrega 2 — Análisis de textos (tópicos, sentimiento, resumen)

Esta entrega amplía el proyecto con **minería de textos** sobre tweets obtenidos por **API (RapidAPI, Twitter/X)**, centrados en **opiniones sobre la película de Michael Jackson** (*Michael*, biopic). El objetivo es explorar qué temas aparecen, cómo se distribuye el sentimiento y generar un **resumen extractivo** del corpus.

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

En el **dashboard** se añaden **árboles de dependencia** (spaCy) solo como apoyo visual del análisis sintáctico, no como salida del extractor.


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
| `extractor.py` | `model_topics`, `analyze_sentiment`, `parse_and_summarize`, stopwords bilingües. |
| `dashboard_rapidapi_tweets.py` | Visualización entrega 2. |
| `data/rapidapi_tweets.csv` | Corpus cacheado de la API. |
