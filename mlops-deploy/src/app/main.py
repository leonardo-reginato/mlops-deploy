from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

# Load modelo ja treinado
lr = pickle.load(open('../../models/ML_lr.sav','rb'))

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)

@app.route('/')
def home():
    return "Minha primeira API."

@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(to='en')
    polaridade = tb_en.sentiment.polarity
    return f"polaridade : {polaridade}"

@app.route('/cotacao/<int:tamanho>')
def cotacao(tamanho):
    preco = lr.predict([[tamanho]])
    return str(preco)

@app.route('/cotacaov2/', methods=['POST'])
def cotacaov2():
    dados = request.get_json()
    colunas = ['tamanho', 'ano', 'garagem']
    dados_in = [dados[col] for col in colunas]
    preco = lr.predict([dados_in])
    return jsonify(preco=preco[0])

@app.route('/cotacaov3/', methods=['POST'])
@basic_auth.required
def cotacaov3():
    dados = request.get_json()
    colunas = ['tamanho', 'ano', 'garagem']
    dados_in = [dados[col] for col in colunas]
    preco = lr.predict([dados_in])
    return jsonify(preco=preco[0])

app.run(debug=True)