import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
from nltk.stem import PorterStemmer
#create an object of class PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn import feature_extraction
porter = PorterStemmer()

def text_preprocess(item):
   lem=WordNetLemmatizer()
   print("In preprocess")
   print(item)
   review=re.sub(r"[^a-zA-Z]",' ', item)
   print(review)
   review=review.lower() 
   print(review)
   review=nltk.sent_tokenize(review)
   print(review)
   words=[nltk.word_tokenize(sent) for sent in review]
   print(words)
   #for i in range(len(words)):
     # words[i]=[lem.lemmatize(word) for word in words[i] if word not in stopwords.words('english')]    
   words=[' '.join(word) for word in words]
   print(words)
   return words
