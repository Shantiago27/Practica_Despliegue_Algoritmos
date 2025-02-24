from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Inicializar la aplicación FastAPI
app = FastAPI()

# Definir modelos de datos
class TextInput(BaseModel):
    text: str

# Endpoint 1: Contar palabras (POST)
@app.post("/contar_palabras")
def contar_palabras(input: TextInput):
    num_palabras = len(input.text.split())
    return {"texto": input.text, "num_palabras": num_palabras}

# Endpoint 2: Calcular factorial (GET)
@app.get("/factorial")
def factorial(n: int):
    def calcular_factorial(x):
        if x == 0 or x == 1:
            return 1
        return x * calcular_factorial(x - 1)
    
    resultado = calcular_factorial(n)
    return {"numero": n, "factorial": resultado}

# Endpoint 3: Pipeline de análisis de sentimientos (POST)
sentiment_pipeline = pipeline("sentiment-analysis")

@app.post("/analizar_sentimiento")
def analizar_sentimiento(input: TextInput):
    resultado = sentiment_pipeline(input.text)
    return {"texto": input.text, "sentimiento": resultado}

# Endpoint 4: Pipeline de generación de texto (POST)
text_generator = pipeline("text-generation", model="gpt2")

@app.post("/generar_texto")
def generar_texto(input: TextInput, max_length: int = 50):
    resultado = text_generator(input.text, max_length=max_length)
    return {"prompt": input.text, "texto_generado": resultado[0]['generated_text']}

# Endpoint 5: Pipeline de traducción de texto (POST)
translator = pipeline("translation_es_to_en", model="Helsinki-NLP/opus-mt-es-en")

@app.post("/traducir")
def traducir(input: TextInput):
    resultado = translator(input.text)
    return {"texto_original": input.text, "traduccion": resultado[0]['translation_text']}

