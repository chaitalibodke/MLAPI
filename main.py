from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd 
from model import train_model

app=FastAPI()

#load the trained model when app starts 
model = train_model()

#define input data struture using pydantic 
class PenguinInput(BaseModel):
    bill_length:float
    flipper_length:float

@app.get("/")
async def read_root():
    return {"message":"Welcome to penguin species prediction API !"}

@app.post("/predict")
async def predict(input_data:PenguinInput):
    features = [[input_data.bill_length,input_data.flipper_length]]
    prediction=model.predict(features)
    species=["Adelie","Chinstrap","Gentoo"][prediction[0]]
    return {"species":species}