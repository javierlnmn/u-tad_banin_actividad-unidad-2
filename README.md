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

El procesamiento se realiza a través de la clase `DataExtractor`, siguiendo un flujo de tres etapas:

### 1. Extracción (ETL)
Se utiliza un patrón **Loader** desacoplado para cargar los datos. El sistema está preparado para manejar archivos grandes mediante el uso de `pandas` y está diseñado para soportar `chunksize` en futuras iteraciones para optimizar el uso de memoria RAM.

### 2. Limpieza y Normalización
La limpieza de texto es fundamental para eliminar el "ruido" de Twitter. El proceso incluye:
* **Case Folding:** Conversión a minúsculas para uniformidad.
* **Sanitización:** Eliminación de URLs (`http/https`) y espacios en blanco extra.
* **Integridad de Hashtags:** Se limpian caracteres especiales pero se preserva el símbolo `#` para permitir una extracción precisa de etiquetas.

### 3. Análisis de Hashtags (Explode Strategy)
Para analizar qué temas son tendencia, el script realiza los siguientes pasos:
* **Regex Extraction:** Identificación de todos los `#tags` dentro del texto.
* **Data Explosion:** Se utiliza el método `.explode()` de Pandas. Si un tweet tiene 3 hashtags, se convierte en 3 filas independientes. Esto permite calcular frecuencias reales sin sesgos.
* **Agregación:** Cálculo de frecuencia global, actividad por usuario y evolución diaria.

## Ejecución con Poetry

Este proyecto utiliza **Poetry** para garantizar que el entorno sea reproducible y que no haya conflictos de librerías.

### Configuración inicial
Si acabas de clonar el repo, instala las dependencias:
```bash
poetry install