from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel

model = load('model.joblib')

app = FastAPI()

@app.get('/calc-residuos')
def calc_garbage(rural: int, urbana: int):
    prediction = model.predict([[rural, urbana]])
    return {
        'prediccion': prediction[0][0]
    }