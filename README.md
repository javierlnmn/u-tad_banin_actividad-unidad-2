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

El procesamiento se realiza a través de la clase `DataExtractor`, siguiendo el siguiente flujo:

### 1. Carga de Datos
Se utiliza un patrón **Loader** desacoplado para cargar los datos usando *pandas*.

### 2. Extracción de Hashtags
Se utiliza una expresión regular para extraer los hashtags del texto.

### 3. Limpieza y Normalización
Se limpia el texto de manera general: mayúsculas a minúsculas, eliminación de URLs, eliminación de caracteres especiales, eliminación de espacios redundantes.

### 4. Extracción de Palabras Clave
Se utiliza una expresión regular para extraer las palabras clave del texto.

### 5. Análisis de Palabras
Se analizan tanto los hashtags como las palabras clave para obtener un análisis más completo, generando un wordcloud.

## Ejecución con Poetry

Este proyecto utiliza **Poetry** para garantizar que el entorno sea reproducible y que no haya conflictos de librerías.

### Configuración inicial
Si acabas de clonar el repo, instala las dependencias:
```bash
poetry install