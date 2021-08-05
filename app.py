
import numpy as np
import preprocess
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
vectorzer=pickle.load(open('trans_vect1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('frontend.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print("In predict function")
    message = request.form['message']
    refined_review = preprocess.text_preprocess(message)
    print(refined_review)
    refined_review=vectorzer.transform(refined_review).toarray()
    print(refined_review.shape)
    prediction = model.predict(refined_review)
    print(prediction)
    if(prediction==0):
        out_txt="Happy with the time spent"
    else:
        out_txt="Dissatisfied with the service"

    return render_template('frontend.html', prediction_text='You are  $ {}'.format(out_txt))

if __name__ == "__main__":
    app.run(debug=True)
