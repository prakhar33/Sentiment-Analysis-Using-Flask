import numpy as np 
from flask import Flask , request , jsonify , render_template
import re
import tensorflow as tf
from keras_preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import json
from tensorflow.compat.v1 import get_default_graph
from keras.models import load_model
import pickle
import keras.backend.tensorflow_backend as tb
#tb._SYMBOLIC_SCOPE.value = True

app = Flask(__name__)


graph = get_default_graph()
with open('model_architecture.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('model_weights.h5')

with open('tokenizer.json' , 'r') as f:
    data = json.load(f)
tokenizer = tokenizer_from_json(data)

def filter_data(text):
    text = re.sub('@[a-zA-Z0-9]+:','',text)
    text = text.lower()
    return text

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/about')
def about():
    return render_template('about.html')    

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    text = request.form['Sentence']
    text=str(text)
    text = [filter_data(text)]
    text = tokenizer.texts_to_sequences(text)
    text = pad_sequences(text, maxlen=300)

    # with graph.as_default():
    # print(text)
    sentiment = model.predict(text , batch_size = 1)
        # return jsonify({'negativity':str(round((1 - sentiment[0][0])*100,3)) , 'positivity':str(round(sentiment[0][0]*100,3))})
    return render_template('index1.html', prediction_text='positivity in your sentiment is {}'.format(str(round(sentiment[0][0]*100))+'%'))

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True  ,threaded=False)

    