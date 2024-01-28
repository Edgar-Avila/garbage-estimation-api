from joblib import load
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

model = load('model.joblib')
scaler = load('scaler.joblib')
population_models = load('population.joblib')
paises_rurales_models = load('paises_rurales.joblib')
paises_urbana_models = load('paises_urbanas.joblib')
score_residuos = 0.9661523416966664

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/calc-residuos')
def calc_garbage(rural: int, urbana: int):
    scaled = scaler.transform([[rural, urbana]])
    prediction = model.predict(scaled)
    return {
        'prediccion': prediction[0][0]
    }

@app.get('/calc-poblacion')
def calc_population(departamento: str, anio: int):
    if departamento not in population_models:
        return {
            'error': 'Departamento no encontrado'
        }
    population_model_urbana = population_models[departamento]['POB_URBANA']['model']
    population_model_rural = population_models[departamento]['POB_RURAL']['model']
    score_urbana = population_models[departamento]['POB_URBANA']['score']
    score_rural = population_models[departamento]['POB_RURAL']['score']
    prediction_urbana = population_model_urbana.predict([[anio]])
    prediction_rural = population_model_rural.predict([[anio]])
    rural = int(prediction_rural[0])
    urbana = int(prediction_urbana[0])
    scaled = scaler.transform([[rural, urbana]])
    prediction_residuos = model.predict(scaled)
    residuos = prediction_residuos[0][0]
    score_total = score_urbana * score_rural * score_residuos
    return {
        'urbana': urbana,
        'rural': rural,
        'residuos': residuos,
        'score_urbana': score_urbana,
        'score_rural': score_rural,
        'score_residuos': score_residuos,
        'score_total': score_total
    }

@app.get('/calc-poblacion/paises')
def calc_population(pais: str, anio: int):
    if (pais not in paises_rurales_models) or (pais not in paises_urbana_models):
        return {
            'error': 'Pais no encontrado'
        }
    population_model_urbana = paises_urbana_models[pais]['model']
    population_model_rural = paises_rurales_models[pais]['model']
    score_urbana = paises_urbana_models[pais]['score']
    score_rural = paises_rurales_models[pais]['score']
    prediction_urbana = population_model_urbana.predict([[anio]])
    prediction_rural = population_model_rural.predict([[anio]])
    rural = int(prediction_rural[0])
    urbana = int(prediction_urbana[0])
    scaled = scaler.transform([[rural, urbana]])
    prediction_residuos = model.predict(scaled)
    residuos = prediction_residuos[0][0]
    score_total = score_urbana * score_residuos
    return {
        'urbana': urbana,
        'rural': rural,
        'residuos': residuos,
        'score_urbana': score_urbana,
        'score_rural': score_rural,
        'score_residuos': score_residuos,
        'score_total': score_total
    }