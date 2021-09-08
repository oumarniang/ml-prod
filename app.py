import sklearn
import joblib
import pandas as pd
import numpy as np
from fonctions_maisons import extraire_la_première_lettre
from flask import Flask, request

print(sklearn.__version__)

# Load Model
pipeline = joblib.load('titanic.model')
#print(pipeline)

# Démarer l'appli
app = Flask('__name__')

## Faire des predicts

@app.route('/predict', methods=['POST'])
def predict():
  #print(request.json)
  #df = pd.DataFrame(pd.read_json(payload, typ='frame', orient='columns'))
  df = pd.DataFrame(request.json)
  résultat = pipeline.predict(df)[0]
  return (str(résultat), 201)
  print(df) 

# Tester l'api
@app.route('/ping', methods=['GET'])
def ping():
  return('pong', 200)

# Page d'accuil
@app.route('/')
def index():
  return "<h1>Bienvenue dans notre API !</h1>"

if __name__=="__main__":
  app.run(host='0.0.0.0')



print(flask.__version__)