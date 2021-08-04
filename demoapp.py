
import numpy as np
import preprocess
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('frontend.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    review = [ x for x in request.form.values()]
    refined_review = text_preprocess(review)
    prediction = model.predict(refined_review)

    if(prediction==0):
        out_txt="Happy with the time spent"
    else:
        out_txt="Dissatisfied with the service"

    return render_template('frontend.html', prediction_text='You are  $ {}'.format(out_txt))

if __name__ == "__main__":
    app.run(debug=True)