from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
import os
from tensorflow.keras.models import load_model
from pydantic import BaseModel

from DRL import Recommender

##########################################################################################
#Load DL Stuff
##########################################################################################
DLdestinations=['Air Terjun Madakaripura',
 'Cukul Sunrise Point',
 'Curug Cikaso',
 'Curug Cipamingkis',
 'Curug Cipendok',
 'Danau Toba',
 'Dunia Fantasi (Dufan)',
 'Grafika Cikole',
 'Green Canyon Pangandaran',
 'Gunung Bromo',
 'Kawah Putih',
 'Kepulauan Seribu',
 'Labuan Bajo',
 'Lombok',
 'Malioboro',
 'Nusa Penida',
 'Pantai Air Manis',
 'Pantai Balekambang',
 'Pantai Gesing',
 'Pantai Kasap',
 'Pantai Klingking',
 'Pantai Pandawa',
 'Pantai Pangandaran',
 'Pantai Santolo',
 'Pantai Sawarna',
 'Pantai Srakung',
 'Pantai Surumanis',
 'Pantai Tampora',
 'Pantai Tanjung Lesung',
 'Pulau Padar',
 'Raja Ampat',
 'Rancabali Glamping',
 'Simpang Lima Gumul',
 'Taman Langit',
 'Taman Laut Bunaken',
 'Taman Mini Indonesia Indah (TMII)',
 'Taman Nasional Bantimurung',
 'Taman Safari Bogor',
 'Taman Safari Pasuruan',
 'Tebing Breksi',
 'Ubud']


class dl_request(BaseModel):
    member : float
    sport: float
    days: float
    time: float
    gender: float
    price: float
    berbelanja: float
    petualang: float
    foto: float
    jalan: float
    selfie: float
    museum: float
    pemandangan: float
    festival: float
    anak: float
    dewasa: float
    lansia: float
    remaja: float
    pertengahan: float


DL_model =load_model("models/deep learning/model/model.h5")

##########################################################################################
#Load DRL Stuff
##########################################################################################
drlRecommender = Recommender()

class pairData(BaseModel):
    destination: str
    like: bool 

class drl_request(BaseModel):
    input: List[pairData]



##########################################################################################
#LOAD AND CONFIGURE FASTAPI
##########################################################################################
app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)



@app.get("/")
async def root():
    return {"message": "Welcome to the ML API!"}

@app.post("/questionnaire")
async def questionnaire(data: dl_request):
    t_x = [[
        data.member,
        data.sport,
        data.days,
        data.time, 
        data.gender,
        data.price, 
        data.berbelanja, 
        data.petualang, 
        data.foto, 
        data.jalan, 
        data.selfie, 
        data.museum, 
        data.pemandangan,
        data.festival,
        data.anak, 
        data.dewasa, 
        data.lansia,
        data.remaja,
        data.pertengahan
    ]]
    prediction = DL_model.predict(t_x)
    result = (-prediction).argsort()[::-1]
    destination_prediction_five = []
    for index in result[0][:5]:
        destination_prediction_five.append(DLdestinations[index])
    return{
        "destinations":destination_prediction_five
    }
    
@app.post("/recommend")
async def recommend(data: drl_request):
    recommendations=drlRecommender.recommend(data.input)

    return {"data":recommendations[:5]}

##########################################################################################
#MAIN
##########################################################################################
if __name__ == "__main__":
	run(app, host="0.0.0.0", port=5001)