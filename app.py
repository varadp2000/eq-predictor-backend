from flask import Flask, jsonify, request, redirect
import keras, numpy as np
from flask_cors import CORS
app = Flask(__name__)
import os
CORS(app)
@app.route('/')
def welcome():
    return "<h1>Welcome to EQ Predictor API</h1><br/>"

@app.route('/predict', methods=['POST', 'GET'])
def Predict():
	if(request.method == 'GET'):
		return redirect("/", code=302)
	if(request.method == 'POST'):
		data = request.get_json()
		lat = data['lat']
		longi = data['longi']
		time = data['time']
		X_train = [[lat, longi, time]]
		X_train = np.asarray(X_train).astype('float32')
		TrainX = np.reshape(X_train,(X_train.shape[0], 1, X_train.shape[1]))
		model = keras.models.load_model('model_lstm.h5')
		res = model.predict(TrainX)
		return jsonify({"magnitude":str(res[0][0]), "depth":str(res[0][1])})



if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT")), debug=True)