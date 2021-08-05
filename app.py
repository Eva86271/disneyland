
import numpy as np
import preprocess
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
cv=pickle.load(open('transform.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('frontend.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    review = [ x for x in request.form.values()]
    refined_review = preprocess.text_preprocess(review[0])
    refined_review=cv.transform(refined_review).toarray()
    prediction = model.predict(refined_review)

    return render_template('frontend.html', prediction_text='You are  $ {}'.format(out_txt))

if __name__ == "__main__":
    app.run(debug=True)