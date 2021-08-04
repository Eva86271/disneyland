import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
#create an object of class PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn import feature_extraction
porter = PorterStemmer()

def text_preprocess(item):
    context=[]
    review=re.sub(r"[^a-zA-Z]",' ', item)
    review=review.lower()
    review=nltk.word_tokenize(review)
    review=[porter.stem(word) for word in review if word not in stopwords.words('english')]
    review=' '.join(review)
    td=feature_extraction.text.TfidfVectorizer()
    transformed_review=td.fit_transform(review)
    return transformed_review
