from flask import Flask, render_template,jsonify, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features = np.array(input_features).reshape(1, -1)
    prediction = model.predict(features)  
    result = prediction[0]

    return render_template('index.html', prediction_text = result)


if __name__ == "__main__":
    app.run(debug=True)