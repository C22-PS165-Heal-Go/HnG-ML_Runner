import pickle
import string
from typing import List
from unittest.util import _MAX_LENGTH
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
import os
from tensorflow.keras.models import load_model
from pydantic import BaseModel
import tensorflow as tf

from RLmodelHelper import Actor, Critic, DRRAveStateRepresentation, PMF
from RLutils.history_buffer import HistoryBuffer

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
DRLpath="models/Deep Reinforcement Learning (DRR)/"

users = pickle.load(open(DRLpath+'dataset_RL/user_id_to_num.pkl', 'rb'))
items = pickle.load(open(DRLpath+'dataset_RL/item_lookup.pkl', 'rb'))
data = np.load(DRLpath+'dataset_RL/data_RL_25000.npy')
data[:, 2] = 0.5 * (data[:, 2] - 3)

## deklarasi model
state_rep_net = DRRAveStateRepresentation(5, 100, 100)
actor_net = Actor(300, 100)
critic_net = Critic(100, 300, 1)
reward_function = PMF(len(users), len(items), 100)
## deklarasi build
state_rep_net.build_model()
actor_net.build_model()
critic_net.build_model()
reward_function.build_model()
## load pretrained
reward_function.load_weights(DRLpath+'trained/pmf_weights/pmf_150')
actor_net.load_weights(DRLpath+'trained/actor_weights/actor_150')
critic_net.load_weights(DRLpath+'trained/critic_weights/critic_150')
state_rep_net.load_weights(DRLpath+'trained/state_rep_weights/state_rep_150')

## sifat user dan item
## untuk candidate item buat dihitung
user_embeddings = tf.convert_to_tensor(reward_function.user_embedding.get_weights()[0])
candidate_items = tf.convert_to_tensor(reward_function.item_embedding.get_weights()[0])

## user buat sementara diacak dalam pemilihan embeddingnya
user_idx = np.random.randint(0, 549)
## ambil embedding user/sifat user
user_emb = user_embeddings[user_idx]

user_reviews = data[data[:, 0] == user_idx]
pos_user_reviews = user_reviews[user_reviews[:, 2] > 0]

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

def getId(dest):
    x=0
    for i in items:
        if items.get(x).lower() == dest.lower():
            return x
        x+=1
    return 0




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
    history_buffer = HistoryBuffer(5)
    for i in range(5):
        emb = candidate_items[int(pos_user_reviews[i, 1])]
        history_buffer.push(tf.identity(tf.stop_gradient(emb)))

    ## list item yang tidak terpilih
    ignored_items = []

    ## variabel buat nampung item hasil rekomendasi
    item_for_recommendation = []

    for item in data.input:
        idx = getId(item.destination)

        if item.like :
            ## ambil embeddingnya terus push ke buffer history
            emb = candidate_items[idx]
            history_buffer.push(tf.identity(tf.stop_gradient(emb)))

            ## statenya hitung
            state = state_rep_net(user_emb, tf.stack(history_buffer.to_list()))

            ## actionnya hitung
            action = actor_net(tf.stop_gradient(state))

            # Perhitungan skor ranking item terbaik dan di-flatten
            ranking_scores = candidate_items @ tf.transpose(action)
            ranking_scores = tf.reshape(ranking_scores, (ranking_scores.shape[0],)).numpy()

            ## ini jaga" biar ga direkomen dua kali
            if len(ignored_items) > 0:
                rec_items = tf.stack(ignored_items).numpy()
            else:
                rec_items = []

            ## nanti di list rankingnya dinegatifin biar di akhir posisinya
            ranking_scores[rec_items] = -float("inf")

            # ambil rekomen itemnya
            rec_item_idx = tf.argmax(ranking_scores).numpy()
            rec_item_emb = candidate_items[rec_item_idx]

            item_for_recommendation.append(items.get(rec_item_idx))

            # Remove new item from future recommendations
            ignored_items.append(tf.convert_to_tensor(rec_item_idx))

        else:
            ## statenya dihitung
            ## karena itemnya gadilike jadi keadaan state masih sama
            state = state_rep_net(user_emb, tf.stack(history_buffer.to_list()))

            ## actionnya hitung
            action = actor_net(tf.stop_gradient(state))

            # Perhitungan skor ranking item terbaik dan di-flatten
            ranking_scores = candidate_items @ tf.transpose(action)
            ranking_scores = tf.reshape(ranking_scores, (ranking_scores.shape[0],)).numpy()

            ## ini jaga" biar ga direkomen dua kali
            if len(ignored_items) > 0:
                rec_items = tf.stack(ignored_items).numpy()
            else:
                rec_items = []

            ## nanti di list rankingnya dinegatifin biar di akhir posisinya
            ranking_scores[rec_items] = -float("inf")

            # ambil rekomen itemnya
            rec_item_idx = tf.argmax(ranking_scores).numpy()
            rec_item_emb = candidate_items[rec_item_idx]

            item_for_recommendation.append(items.get(rec_item_idx))

            # Remove new item from future recommendations
            ignored_items.append(tf.convert_to_tensor(rec_item_idx))

    return {"data":item_for_recommendation}

##########################################################################################
#MAIN
##########################################################################################
if __name__ == "__main__":
	run(app, host="0.0.0.0", port=5001)