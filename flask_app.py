from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
from helper_functions import shortenFEN, unflipFEN
import helper_image_loading
import chessboard_finder
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model (using the same approach from the class you provided)
def load_graph(frozen_graph_filepath):
    with tf.io.gfile.GFile(frozen_graph_filepath, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="tcb")
    return graph

class ChessboardPredictor:
    def __init__(self, frozen_graph_path='saved_models/frozen_graph.pb'):
        graph = load_graph(frozen_graph_path)
        self.sess = tf.compat.v1.Session(graph=graph)
        self.x = graph.get_tensor_by_name('tcb/Input:0')
        self.keep_prob = graph.get_tensor_by_name('tcb/KeepProb:0')
        self.prediction = graph.get_tensor_by_name('tcb/prediction:0')
        self.probabilities = graph.get_tensor_by_name('tcb/probabilities:0')

    def getPrediction(self, tiles):
        validation_set = np.swapaxes(np.reshape(tiles, [32*32, 64]), 0, 1)
        guess_prob, guessed = self.sess.run([self.probabilities, self.prediction], feed_dict={self.x: validation_set, self.keep_prob: 1.0})
        tile_certainties = np.array(list(map(lambda x: x[0][x[1]], zip(guess_prob, guessed)))).reshape([8,8])[::-1,:]
        labelIndex2Name = lambda label_index: ' KQRBNPkqrbnp'[label_index]
        pieceNames = list(map(lambda k: '1' if k == 0 else labelIndex2Name(k), guessed))
        fen = '/'.join([''.join(pieceNames[i*8:(i+1)*8]) for i in reversed(range(8))])
        return fen, tile_certainties

# HTTP API endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded image
    filepath = os.path.join("/tmp", secure_filename(file.filename))
    file.save(filepath)

    # Load and process the image
    img = helper_image_loading.loadImageFromPath(filepath)
    tiles, corners = chessboard_finder.findGrayscaleTilesInImage(img)
    if tiles is None:
        return jsonify({'error': 'Could not find chessboard in image'}), 400

    # Make prediction
    predictor = ChessboardPredictor()
    fen, tile_certainties = predictor.getPrediction(tiles)
    predictor.sess.close()

    
    short_fen = shortenFEN(fen)
    certainty = tile_certainties.min()
    # weird conversion back and forth to get float32
    certainty = np.float32(certainty)  
    certainty = float(certainty)

    return jsonify({'fen': short_fen +" w - - 0 1", 'certainty': certainty})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
