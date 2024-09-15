import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Vamos a crear algunas funciones que nos serán útiles para procesar el texto
# 1) Identificamos si el SERNAC fue mencionado
def mencion_sernac(words):
    sernac = "No"
    for word in words:
        if word.lower().find("sernac") != -1:
            sernac = "Si"
    return sernac

# 2) Quitamos las menciones y los links
def quitar_menciones(words):
    palabras = []
    for word in words:
        if ((word.startswith('@')) & (len(word) > 1)):
            palabra = ''
        elif word.startswith('http'):
            palabra = ''
        else:
            palabra = word
        palabras.append(palabra)
        texto = ' '.join(palabras)
    return texto

# 3) Obtenemos el sentimiento del tweet procesado
def obtener_sentimiento(df):
    inputs = tokenizer(df, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = modelo(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probs).item()
    
    # Mapeo de los resultados a sentimiento (0: muy negativo, 4: muy positivo)
    sentiment_map = {0: 'Muy negativo', 1: 'Negativo', 2: 'Neutral', 3: 'Positivo', 4: 'Muy positivo'}
    return sentiment_map[sentiment]


# Cargamos el dataset
df_tweets = pd.read_csv("Tweets.csv", sep="|")

# Establecemos si el tweet viene desde un usuario, desde la empresa o del SERNAC
df_tweets.loc[df_tweets['username'] == 'RipleyChile', 'origen'] = 'Empresa'
df_tweets.loc[df_tweets['username'] == 'SERNAC', 'origen'] = 'SERNAC'
df_tweets["origen"].fillna("Usuario", inplace= True)

# Transformamos la fecha
df_tweets["fecha"] = df_tweets["date"].astype(str)
df_tweets["fecha"] = pd.to_datetime(df_tweets["fecha"].str[:10], format="%Y-%m-%d")
df_tweets.drop(columns='date', inplace= True)

# Cargamos el modelo pre-entrenado y el tokenizador
nombre_modelo = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)
modelo = AutoModelForSequenceClassification.from_pretrained(nombre_modelo)

# Procesamos el texto y luego obtenemos las menciones al SERNAC y el sentimiento del tweet

df_tweets["texto_procesado"] = df_tweets["content"].astype(str).str.lower()
df_tweets["texto_procesado"] = df_tweets["texto_procesado"].str.split(' ')

df_tweets["Sernac"] = df_tweets["texto_procesado"].apply(mencion_sernac)
df_tweets["texto_procesado"] = df_tweets['texto_procesado'].apply(quitar_menciones)
df_tweets['Sentimiento'] = df_tweets['texto_procesado'].apply(obtener_sentimiento)

# Guardamos el archivo con los resultados
df_tweets.to_csv('Tweets_con_sentimiento.csv', index=False)