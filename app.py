from flask import Flask, request, jsonify
from scipy.sparse import coo_matrix
import numpy as np
import tensorflow as tf
import os
app = Flask(__name__)

@app.errorhandler(404)
def not_found(e):
  return jsonify({"error": "Page not found error"})
  
@app.errorhandler(405)
def method_not_allowed(e):
  return jsonify({"error": "Method Not Allowed"})

@app.route('/')
def inform():
  return jsonify({"error": "Call /predict/<model-name> in post method with body with user prefernce to get prediction"})

@app.route('/predict/<string:model_name>', methods=['POST'])
def Prediction(model_name):
  json = request.get_json()

  multvae = tf.keras.models.load_model('models/multvae')
  if model_name=='multvae':
    vae = multvae
  else:
    return jsonify({"error":"Model Not Found"})
  k = 10
  
  preferred_movies = json['preferred_movies']
  size = len(preferred_movies)
  data = np.ones((size)).astype('int')
  row = np.zeros((size)).astype('int')
  col = np.array(preferred_movies)

  user_matrix = np.array(coo_matrix((data, (row, col)), shape=( 1,62000)).todense())
  
  reconstructed_matrix = vae.decoder(vae.encoder(user_matrix)).numpy()

  sorted_ratings = reconstructed_matrix[0].tolist()

  top_predicted_movies_idx = sorted(range(len(sorted_ratings)), key=lambda i: sorted_ratings[i])[-k:]

  print(top_predicted_movies_idx)
  return jsonify({"error": "", "predictions": top_predicted_movies_idx})


if __name__=='__main__':

  app.run()