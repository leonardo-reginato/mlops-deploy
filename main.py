from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

#df = pd.read_csv('https://raw.githubusercontent.com/alura-cursos/1576-mlops-machine-learning/aula-5/casas.csv')
#
## cols = ['tamanho', 'preco']
## df = df[cols]
#
#x = df.drop('preco',axis=1)
#y = df['preco']
#
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#
#lr = LinearRegression()
#lr.fit(x_train, y_train)

# Load modelo ja treinado
lr = pickle.load(open('ML_lr.sav','rb'))

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'leo'
app.config['BASIC_AUTH_PASSWORD'] = 'alura'

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