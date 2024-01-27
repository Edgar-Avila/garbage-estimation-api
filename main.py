from joblib import load
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

model = load('model.joblib')
scaler = load('scaler.joblib')
population_models = load('population.joblib')
population_scaler = population_models['scaler']

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
    scaled = population_scaler.transform([[anio]])
    prediction_urbana = population_model_urbana.predict(scaled)
    prediction_rural = population_model_rural.predict(scaled)
    rural = int(prediction_rural[0])
    urbana = int(prediction_urbana[0])
    scaled_residuos = scaler.transform([[rural, urbana]])
    prediction_residuos = model.predict(scaled_residuos)
    residuos = prediction_residuos[0][0]
    return {
        'urbana': urbana,
        'rural': rural,
        'residuos': residuos
    }