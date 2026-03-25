# Bitcoin Tweets Analysis (Kaggle Dataset)

Este proyecto implementa un pipeline de datos para procesar, limpiar y analizar tendencias en la conversación de Twitter sobre Bitcoin, utilizando el dataset masivo de Kaggle.


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

El flujo lo concentra la clase `DataExtractor`. El punto de entrada es `generate_hashtag_wordcloud()`, que por dentro se apoya en `analytics_hashtags_extended()`:

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

## Ejecución con Poetry

Este proyecto utiliza **Poetry** para garantizar que el entorno sea reproducible y que no haya conflictos de librerías.

### Configuración inicial
Si acabas de clonar el repo, instala las dependencias:
```bash
poetry install
```

Para ejecutar el proyecto:

```bash
poetry run python main.py
```

## Dashboard interactivo

App **Streamlit** (`dashboard.py`) para ver el mismo análisis de forma interactiva: métricas, gráficos (Plotly), tablas y wordcloud. Tras instalar dependencias:

```bash
poetry run streamlit run dashboard.py
```
