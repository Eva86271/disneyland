#!/usr/bin/env python
# coding: utf-8

# ## DisneyLand Analysis

# In[311]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# In[312]:


disney_data=pd.read_csv("C:/Users/subha/Downloads/reviews/DisneylandReviews.csv",encoding='latin-1')


# In[313]:


disney_data.head()


# In[314]:


disney_data.shape


# In[315]:


disney_data.info()


# In[316]:


for i in disney_data.columns:
    print(disney_data[i].unique())


# In[317]:


Category=[]
for i in range(len(disney_data['Rating'])):
    if(disney_data.loc[i,'Rating']== 1):
        Category.append("Dissatisfied")
    elif(disney_data.loc[i,'Rating']== 2):
        Category.append("Poor")
    elif(disney_data.loc[i,'Rating']== 3):
        Category.append("Average")
    elif(disney_data.loc[i,'Rating']== 4):
        Category.append("Good")
    elif(disney_data.loc[i,'Rating']== 5):
        Category.append("Excellent")


# In[318]:


disney_data["Category"]=Category


# In[319]:


Nature=[]
for i in range(len(disney_data['Rating'])):
    if(disney_data.loc[i,'Rating']>2):
        Nature.append("Happy with the time")   
    elif(disney_data.loc[i,'Rating']<=2):
        Nature.append("Dissatisfied")
disney_data["Nature"]=Nature


# In[320]:


fig, ax = plt.subplots(figsize=(10,10))
disney_data["Category"].value_counts().head(10).plot(kind="bar",color="green")
plt.show()


# In[321]:


fig, ax = plt.subplots(figsize=(10,10))
disney_data["Reviewer_Location"].value_counts().head(10).plot(kind="bar",color="orange")
plt.show()


# In[322]:


fig, ax = plt.subplots(figsize=(10,10))
splot=disney_data.groupby(disney_data["Branch"])["Category"].value_counts().plot(kind="bar",color="green")
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
plt.show()


# In[323]:


fig, ax = plt.subplots(figsize=(10,10))
splot=disney_data['Branch'].value_counts().plot(kind="bar",color="blue")
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
plt.show()


# In[324]:


fig, ax = plt.subplots(figsize=(10,10))
splot=disney_data.groupby(disney_data["Branch"])["Rating"].mean().plot(kind="bar",color="green")
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
plt.show()
fig, ax = plt.subplots(figsize=(10,10))
splot=disney_data.groupby(disney_data["Branch"])["Rating"].count().plot(kind="bar",color="red")
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
plt.show()


# In[325]:


sns.catplot(x="Rating", hue="Category", col="Branch",data=disney_data, kind="count",height=4, aspect=.7)


# In[326]:



year=disney_data['Year_Month'].apply(lambda x: x.split('-')[0])
print(year)


# In[327]:


disney_data["Year"]=year
indexcount=disney_data[disney_data["Year"]=='missing'].index
print(indexcount)


# In[328]:


new_data=disney_data.drop(indexcount)
new_data.reset_index(inplace=True)
new_data["Year"]=new_data['Year_Month'].apply(lambda x: x.split("-")[0])
new_data["Month"]=new_data['Year_Month'].apply(lambda x: x.split("-")[1])


# In[329]:


new_data.head()


# In[330]:


new_data.drop('Year_Month',axis=1,inplace=True)


# In[331]:


### How many Visit 
sns.catplot(x="Branch", hue="Year",
                data=new_data, kind="count",
                height=8);


# In[332]:


### How many Visit 
sns.catplot(x="Branch", hue="Reviewer_Location",
                data=new_data, kind="count",
                height=8);


# In[333]:


import nltk
from gensim.models import Word2Vec


# In[334]:


from nltk.corpus import stopwords
import re


# In[355]:


disney_data_text=disney_data[['Review_Text','Nature']]


# In[356]:


disney_data_text.head()


# In[357]:


disney_data_text['Nature'].value_counts()


# In[358]:


df1=disney_data_text[disney_data_text['Nature']=="Happy with the time"].sample(frac=0.88, random_state=42)
disney_data_text=disney_data_text.drop(df1.index)


# In[359]:


disney_data_text['Nature'].value_counts()


# In[ ]:





# In[360]:


context=[]
disney_data_text.reset_index(inplace=True)
for i in range(len(disney_data_text['Review_Text'])):
    review=re.sub(r"[^a-zA-Z]",' ', disney_data_text.loc[i,'Review_Text'])
    review=review.lower()
    context.append(review)
    
    


# In[361]:


print(context[0])


# In[362]:


for i in range(len(context)):
    context[i]=nltk.word_tokenize(context[i])


# In[363]:


print(context[0])


# In[364]:


from nltk.stem import PorterStemmer
#create an object of class PorterStemmer
from nltk.stem import WordNetLemmatizer
porter = PorterStemmer()
for i in range(len(context)):
    context[i]=[porter.stem(word) for word in context[i] if word not in stopwords.words('english')]
    context[i]=' '.join(context[i])


# In[365]:


#context[i]=' '.join(context[i])
print(context[0])
    


# In[366]:


from sklearn import preprocessing
lb=preprocessing.LabelEncoder()
disney_data_text['Nature']=lb.fit_transform(disney_data_text['Nature'])


# In[367]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(context,disney_data_text['Nature'],test_size=0.20,random_state=42)


# In[368]:


#Tokenization
from sklearn import feature_extraction
td=feature_extraction.text.TfidfVectorizer()


# In[369]:



y=X_train.index
wordcloud1 = WordCloud().generate(X_train[0])
wordcloud2 = WordCloud().generate(X_train[1])
#words used by different twitters 
print(X_train[0])
plt.imshow(wordcloud1, interpolation='bilinear')
plt.axis("Off")
plt.show()
print(X_train[1])
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("Off")
plt.show()


# In[370]:


word_train=td.fit_transform(X_train)


# In[371]:


word_test=td.transform(X_test)


# In[372]:


print("Shape of the train data")
word_train.shape


# In[373]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

text_model_tree=RandomForestClassifier(n_estimators=200).fit(word_train,Y_train)
text_model = MultinomialNB().fit(word_train,Y_train)


# In[374]:


y_predict=text_model.predict(word_test)
print(np.mean(y_predict==Y_test))


# In[375]:


y_predict=text_model_tree.predict(word_test)
print(np.mean(y_predict==Y_test))


# In[376]:


disney_data_text['Nature'].unique()


# In[379]:


from sklearn.linear_model import LogisticRegression
log_class=LogisticRegression().fit(word_train,Y_train)
y_predict=log_class.predict(word_test)
print(np.mean(y_predict==Y_test))


# In[380]:


import pickle
pickle.dump(log_class,open("model.pkl",'wb'))
model = pickle.load(open('model.pkl','rb'))


# In[ ]:




