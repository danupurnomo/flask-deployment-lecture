from flask import Flask, jsonify, request
import numpy as np
import base64
from tensorflow.keras.preprocessing import image

# App Initialization
app = Flask(__name__)

# Load Sequential Model
from tensorflow.keras.models import load_model
model = load_model('rock-paper-scissor-model.h5')

@app.route("/")
def home():
    return "<h1>It Works!</h1>"

@app.route('/predict', methods=['POST'])
def register_new():
    args = request.json
    user_image = np.array(args['user_image'])

    # Change Image Dimension from (height, width, channel) to (1, height, width, channel)
    user_image = np.expand_dims(user_image, axis=0)
    user_image = np.vstack([user_image])
    
    prediction = model.predict(user_image)
    result_max_proba = prediction.argmax(axis=-1)[0]

    class_names = ['paper', 'rock', 'scissors']
    label_names = class_names[result_max_proba]
    
    print('[DEBUG] Result : ', prediction, result_max_proba, label_names)
    
    response = jsonify(
      result = str(result_max_proba), 
      label_names = label_names)

    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0')