import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
#create an object of class PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn import feature_extraction
porter = PorterStemmer()
def text_preprocess(item):
   lem=WordNetLemmatizer()
   review=re.sub(r"[^a-zA-Z]",' ', item)
   review=review.lower() 
   review=nltk.sent_tokenize(review)   
   words=[nltk.word_tokenize(sent) for sent in review]
   #for i in range(len(words)):
      #words[i]=[porter.stem(word) for word in words[i] if word not in stopwords.words('english')]    
   words=[' '.join(word) for word in words]
   return words
