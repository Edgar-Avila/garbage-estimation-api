from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

model = load('model.joblib')
scaler = load('scaler.joblib')

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