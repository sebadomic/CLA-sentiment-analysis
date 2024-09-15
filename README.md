# Análisis de Sentimiento y Menciones al SERNAC en Tweets

Este repositorio contiene un script en Python que procesa tweets para identificar menciones al SERNAC, elimina menciones y enlaces del texto, y obtiene el sentimiento del tweet utilizando un modelo preentrenado de BERT en hugging-face, llamado nlptown/bert-base-multilingual-uncased-sentiment. Puedes encontrar más información sobre este modelo en **https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment**


## Descripción

El script realiza los siguientes pasos:

1. **Identificación de menciones al SERNAC**: Revisa el contenido del tweet para ver si hay menciones a la institución "SERNAC".
2. **Eliminación de menciones y enlaces**: Limpia el texto eliminando menciones a otros usuarios y cualquier enlace.
3. **Análisis de sentimiento**: Usa un modelo preentrenado de BERT para analizar el sentimiento de cada tweet. Los posibles resultados son: Muy negativo, Negativo, Neutral, Positivo, Muy positivo.
4. **Clasificación del origen**: Clasifica si el tweet fue publicado por un usuario, la empresa (`RipleyChile`) o el SERNAC.
5. **Transformación de fecha**: Convierte el formato de fecha del tweet a un formato estándar.
6. **Guardado de resultados**: Los resultados finales, que incluyen el origen del tweet, las menciones al SERNAC y el sentimiento, se guardan en un archivo CSV.

## Requisitos

Para ejecutar el código, es necesario tener instalados los siguientes paquetes:

- `pandas`: Para manipulación y análisis de datos.
- `transformers`: Para cargar el modelo preentrenado de BERT.
- `torch`: Para manejar el modelo de BERT y realizar inferencias de sentimiento.

Puedes instalarlos utilizando pip:
```bash
pip install pandas transformers torch
```
# El archivo de entrada debe ser un archivo Tweets.csv que contiene las siguientes columnas que se utilizarán:

1. **username**: Nombre del usuario que publicó el tweet.
2. **content**: El texto del tweet.
3. **date**: Fecha de publicación del tweet (en formato de cadena de texto).

# El script procesará los tweets y generará un archivo Tweets_con_sentimiento.csv con las siguientes columnas adicionales:

1. **Sernac**: Indica si el usuario menciona al SERNAC
2. **Sentimiento**: Cuyos posibles valores son [Muy negativo, Negativo, Neutral, Positivo, Muy positivo].

También se adhiere un jupyter noteebook en el que se hace el análisis de los resultados obtenidos para luego crear el script para la extracción y procesamiento de los tweets