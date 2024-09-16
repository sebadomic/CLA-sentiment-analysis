import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Vamos a crear algunas funciones que nos serán útiles para procesar el texto
# 1) Identificamos si el gobierno fue mencionado
entidades = ['@IPSChile','@MintrabChile','@GobiernodeChile','@MinDesarrollo','@previsionsocial',
                 '@subse_ssociales','@SENAMAGOB','@IPSValparaiso','@MDSValparaiso','@ChileAtiende','@IPSChile',
            '@RegCivil_Chile','@GobDigitalCL']

def mencion_gobierno(texto):
    for entidad in entidades:
        if entidad.lower() in texto.lower():
            return "Si"
    return "No"

# 2) Quitamos las menciones y los links
def quitar_menciones(words):
    palabras = []
    for word in words:
        if ((word.startswith('@')) & (len(word) > 1)):
            palabra = ''
        elif word.startswith('http'):
            palabra = ''
        elif word == 'rt':
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

# 4) Reemplazamos los saltos de línea 
def reemplazar_saltos_de_linea(texto):
    return texto.replace('\n', ' ')

# Cargamos el dataset
df_tweets = pd.read_csv("dataset_Caso_2.csv")

# Cargamos el modelo pre-entrenado y el tokenizador
nombre_modelo = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)
modelo = AutoModelForSequenceClassification.from_pretrained(nombre_modelo)

# Procesamos el texto y luego obtenemos las menciones y el sentimiento del tweet

df_tweets["texto_procesado"] = df_tweets["texto_tweet"].astype(str).str.lower()
df_tweets["texto_procesado"] = df_tweets["texto_procesado"].apply(reemplazar_saltos_de_linea)
df_tweets["texto_procesado"] = df_tweets["texto_procesado"].str.split(' ')
df_tweets["Mencion_Gobierno"] = df_tweets["texto_tweet"].apply(mencion_gobierno)
df_tweets["texto_procesado"] = df_tweets['texto_procesado'].apply(quitar_menciones)
df_tweets['Sentimiento'] = df_tweets['texto_procesado'].apply(obtener_sentimiento)

# Guardamos el archivo con los resultados
df_tweets.to_csv('data_caso_2_con_sentimiento.csv', index=False)