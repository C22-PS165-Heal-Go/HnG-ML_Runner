from numpy import number
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
import os
# import tensorflow as tf
from tensorflow.keras.models import load_model
from pydantic import BaseModel


destinations=['Air Terjun Madakaripura, Jawa Timur',
 'Cukul Sunrise Point, Jawa Barat',
 'Curug Cikaso, Jawa Barat',
 'Curug Cipamingkis, Jawa Barat',
 'Curug Cipendok, Jawa Tengah',
 'Danau Toba, Sumatera Utara',
 'Dunia Fantasi (Dufan), Jakarta',
 'Grafika Cikole, Jawa Barat',
 'Green Canyon Pangandaran, Jawa Barat',
 'Gunung Bromo, Jawa Timur',
 'Kawah Putih, Jawa Barat',
 'Kepulauan Seribu, Jakarta',
 'Labuan Bajo, NTT',
 'Lombok, NTB',
 'Malioboro, Yogyakarta',
 'Nusa Penida, Bali',
 'Pantai Air Manis, Sumatera Barat',
 'Pantai Balekambang, Jawa Timur',
 'Pantai Gesing, Yogyakarta',
 'Pantai Kasap, Jawa Timur',
 'Pantai Klingking, Bali',
 'Pantai Pandawa, Bali',
 'Pantai Pangandaran, Jawa Barat',
 'Pantai Santolo, Jawa Barat',
 'Pantai Sawarna, Banten',
 'Pantai Srakung, Yogyakarta',
 'Pantai Surumanis, Jawa Tengah',
 'Pantai Tampora, Jawa Timur',
 'Pantai Tanjung Lesung, Banten ',
 'Pulau Padar, NTT',
 'Raja Ampat, Papua Barat',
 'Rancabali Glamping, Jawa Barat',
 'Simpang Lima Gumul, Jawa Timur ',
 'Taman Langit, Jawa Barat',
 'Taman Laut Bunaken, Sulawesi Utara',
 'Taman Mini Indonesia Indah (TMII), Jakarta',
 'Taman Nasional Bantimurung, Sulawesi Selatan',
 'Taman Safari Bogor, Jawa Barat',
 'Taman Safari Pasuruan, Jawa Timur',
 'Tebing Breksi, Yogyakarta',
 'Ubud, Bali']


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

@app.post("/test")
async def test(data: dl_request):
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
        destination_prediction_five.append(destinations[index])
    return{
        "destinations":destination_prediction_five
    }
    
if __name__ == "__main__":
	run(app, host="0.0.0.0", port=5001)