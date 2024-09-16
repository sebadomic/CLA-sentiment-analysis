# Análisis de Sentimiento y Menciones al SERNAC en Tweets

Este repositorio contiene un script en Python que procesa tweets para identificar menciones al gobierno, elimina menciones y enlaces del texto, y obtiene el sentimiento del tweet utilizando un modelo preentrenado de BERT en hugging-face, llamado nlptown/bert-base-multilingual-uncased-sentiment. Puedes encontrar más información sobre este modelo en **https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment**


## Descripción

El script realiza los siguientes pasos:

1. **Identificación de menciones al GOBIERNO**: Revisa el contenido del tweet para ver si hay menciones a las instituciones del país.
2. **Eliminación de menciones y enlaces**: Limpia el texto eliminando menciones a otros usuarios y cualquier enlace.
3. **Análisis de sentimiento**: Usa un modelo preentrenado de BERT para analizar el sentimiento de cada tweet. Los posibles resultados son: Muy negativo, Negativo, Neutral, Positivo, Muy positivo.
4. **Guardado de resultados**: Los resultados finales, que incluyen el origen del tweet, las menciones al SERNAC y el sentimiento, se guardan en un archivo CSV.

## Requisitos

Para ejecutar el código, es necesario tener instalados los siguientes paquetes:

- `pandas`: Para manipulación y análisis de datos.
- `transformers`: Para cargar el modelo preentrenado de BERT.
- `torch`: Para manejar el modelo de BERT y realizar inferencias de sentimiento.

Puedes instalarlos utilizando pip:
```bash
pip install pandas transformers torch
```
# El archivo de entrada debe ser un archivo dataset_Caso_2.csv que contiene las siguientes columnas que se utilizarán:

1. **texto_tweet**: Nombre del usuario que publicó el tweet.

# El script procesará los tweets y generará un archivo data_caso_2_con_sentimiento.csv con las siguientes columnas adicionales:

1. **texto_procesado**: muestra el texto resultante luego de su procesamiento
2. **Mencion_Gobierno**: Mueestra si dentro del tweet se menciona a alguna institución gubernamental del país..
3. **Sentimiento**: Cuyos posibles valores son [Muy negativo, Negativo, Neutral, Positivo, Muy positivo].

También se adhiere un jupyter noteebook "Casi_2_CLA_Sebastian_Ramirez.ipynb" en el que se hace el análisis de los resultados obtenidos para luego crear el script para la carga y procesamiento de los tweets.