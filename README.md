# Bitcoin Tweets Analysis (Kaggle Dataset)

Este proyecto implementa un pipeline de datos para procesar, limpiar y analizar tendencias en la conversación de Twitter sobre Bitcoin, utilizando el dataset de Kaggle [Bitcoin Tweets](https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets).


## Fuente de Datos
* **Dataset:** [Bitcoin Tweets](https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets) de Kaggle.
* **Volumen:** ~170,000 registros.
* **Campos**:
  - `user_name`
  - `user_location`
  - `user_description`
  - `user_created`
  - `user_followers`
  - `user_friends`
  - `user_favourites`
  - `user_verified`
  - `date`
  - `text`
  - `hashtags`
  - `source`
  - `is_retweet`


## Metodología de Trabajo

El flujo lo concentra la clase `DataExtractor` (`extractor.py`). El punto de entrada es `generate_hashtag_wordcloud()`, que por dentro se apoya en `analytics_hashtags_extended()`:

### 1. Carga de datos
Patrón **Loader** (p. ej. `KaggleLoader`) y lectura con *pandas*.

### 2. Limpieza del texto
`clean_text`: minúsculas, URLs fuera, emojis y ruido fuera, espacios normalizados. **No se elimina el carácter `#`**, para poder localizar hashtags en el texto ya normalizado.

### 3. Extracción de hashtags
Expresión regular (`#` + palabra) sobre el texto limpio; una lista de hashtags por fila.

### 4. Agregación y métricas
Las listas se *explotan* en filas; se calculan frecuencias globales, por usuario (`user_name`) y por día (`date`).

### 5. Visualización
**Wordcloud** a partir de las frecuencias globales (`wordcloud` + matplotlib).

## Entorno e instalación

Desde la raíz del proyecto (recomendado: entorno virtual):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Windows, activa el entorno con `.venv\Scripts\activate`.

## Ejecución de la CLI

La línea de comandos se define en `cli.py`. Por defecto se usa el loader **csv** y la ruta `data/Bitcoin_tweets_dataset_2.csv`.

```bash
python main.py
```

Ayuda y opciones:

```bash
python main.py -h
```

| Opción | Descripción |
|--------|-------------|
| `--loader {csv,kaggle,json}` | Origen de datos. `json` no está implementado y termina con código de salida 2. |
| `--csv-path RUTA` | CSV local (solo tiene efecto con `--loader csv`). |
| `--kaggle-dataset SLUG` | Dataset en Kaggle, p. ej. `kaushiksuresh147/bitcoin-tweets` (con `--loader kaggle`). |
| `--kaggle-file NOMBRE` | Fichero dentro del dataset (con `--loader kaggle`). |
| `--export` | Escribe `cleaned_dataset.csv` (dataset limpio) en `--output-dir`. |
| `--output-dir` | Carpeta para `--export` (por defecto `output`). |

Ejemplos:

```bash
# CSV local explícito
python main.py --loader csv --csv-path data/Bitcoin_tweets_dataset_2.csv

# Descarga / carga vía Kaggle (requiere credenciales Kaggle configuradas)
python main.py --loader kaggle

# Dataset y fichero concretos en Kaggle
python main.py --loader kaggle --kaggle-dataset kaushiksuresh147/bitcoin-tweets --kaggle-file Bitcoin_tweets_dataset_2.csv

# Mismo flujo con datos de Kaggle y exportación
python main.py --loader kaggle --export

# Exportar en otra carpeta
python main.py --loader csv --export --output-dir mi_salida
```

## Dashboard interactivo

Hay dos apps de Streamlit llamadas `dashboards` (ejecutar desde la raíz del repo):

- **`dashboard_bitstream.py`**: corpus Bitcoin en Kaggle (hashtags + LDA, sentimiento, resumen).
- **`dashboard_rapidapi_tweets.py`**: mismos paneles con tweets vía RapidAPI o `data/rapidapi_tweets.csv`.

```bash
streamlit run dashboard_bitstream.py
streamlit run dashboard_rapidapi_tweets.py
```
