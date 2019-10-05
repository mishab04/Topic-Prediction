# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 11:31:10 2018

@author: Suresukumar.Gk
"""


import os

os.chdir('C:/Users/366461/Documents/TCS Internal/Disease')

# Importing Libraries 
import numpy as np 
import pandas as pd 
import pickle

# Import dataset 
dataset = pd.read_csv('Disease Classification-Disease_Treatment.csv',encoding = 'latin-1') 

# library to clean data 
import re 

#changing to lowercase the text
dataset['Content'] = dataset['Content'].apply(lambda x: x.lower())


from nltk.corpus import stopwords

stop_words = stopwords.words("english")
newstopwords = ['treatment', 'take', 'make', 'keep', 'many', 'using', 'come', 'number',
                  'useful', 'well', 'disease', 'good', 'feel', 'work', 'important',
                  'method', 'first', 'people', 'part', 'form', 'patient', 'cold', 'time',
                  'need', 'help', 'include', 'even', 'three', 'times', 'often', 'solution',
                  'case', 'possible', 'used', 'case', 'effect', 'effective', 'necessary',
                  'half', 'hour', 'helps', 'leave', 'increase', 'taking', 'apply', 'found',
                  'person', 'begin', 'side', 'place', 'year', 'days', 'another', 'necessary',
                  'problem', 'result', 'best', 'effects', 'much', 'long', 'following',
                  'drop', 'product', 'solution', 'best', 'contain', 'better', 'reduce',
                  'condition', 'years', 'every', 'common', 'start', 'allow', 'problem',
                  'usually', 'found', 'addition', 'reduce', 'without', 'month', 'daily',
                  'must', 'amount', 'known', 'called', 'less', 'composition', 'others',
                  'several', 'hand', 'especially', 'become', 'type', 'enough', 'example',
                  'little', 'give', 'lead', 'small', 'mean', 'occur', 'benefit', 'home',
                  'contains', 'point', 'remove', 'back', 'thing', 'leaves']

stop_words.extend(newstopwords)

'''
Defining function to remove stopwords
'''

def remove_stopwords(text):
    filter_word = [word for word in text.split() if word not in stop_words]
    filter_text = ' '.join(filter_word)
    return filter_text

#removing stopwords
dataset['Content'] = dataset['Content'].apply(lambda x: remove_stopwords(x))
#removing punctuation from content
dataset['Content'] = dataset['Content'].apply(lambda x: re.sub('[^\w\s]','',x))
#removing numbers from content
dataset['Content'] = dataset['Content'].apply(lambda x: re.sub('[1234567890]','',x))
#replacing white space from content
dataset['Content'] = dataset['Content'].apply(lambda x: re.sub('\s+',' ',x))

from nltk.stem.wordnet import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

'''
Defining function for Lemmatization
'''

def word_lemmatize(text):
    lemma_word = [wordnet_lemmatizer.lemmatize(word) for word in text.split()]
    lemma_text = ' '.join(lemma_word)
    return lemma_text

dataset['Content'] = dataset['Content'].apply(lambda x: word_lemmatize(x))

'''
Defining function for Lemmatization for using spacy
'''

import spacy

nlp = spacy.load('en_core_web_sm')

def spacy_lemmatize(text):
    spacy_lemma_word = [w.lemma_ for w in nlp(text)]
    lemma_text = ' '.join(spacy_lemma_word)
    return lemma_text

dataset['Content'] = dataset['Content'].apply(lambda x: spacy_lemmatize(x))

'''
Defining function to remove short words
'''

def remove_shortwords(text):
    short_word = [word for word in text.split() if len(word) > 3]
    short_text = ' '.join(short_word)
    return short_text

#removing stopwords
dataset['Content'] = dataset['Content'].apply(lambda x: remove_shortwords(x))
#removing blank lines from data
dataset = dataset[dataset['Content'].notnull()]

all_words = ' '.join([text for text in dataset['Content']])

import matplotlib.pyplot as plt
from wordcloud import WordCloud
wordcloud = WordCloud(width=1000, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#using countvectorizer for creating dtm

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_df=0.95, min_df=2)

dtm = cv.fit_transform(dataset['Content'])

# now pickle
pickle.dump(cv, open("vector.pickel", "wb"))

dtm

#Using LDA for topic modelling

from sklearn.decomposition import LatentDirichletAllocation   

LDA = LatentDirichletAllocation(n_components=6,random_state=42)

LDA.fit(dtm)

# now pickle
pickle.dump(LDA, open("model.pickel", "wb"))


#Getting top 15 words for each topic

x = {}

for index,topic in enumerate(LDA.components_):
    y = {index:[cv.get_feature_names()[i] for i in topic.argsort()[-15:]]}
    x.update(y)

topics_words = pd.DataFrame.from_dict(x)

topics_words.to_csv('topic_words.csv')

#Assigning topics to each row

topic_results = LDA.transform(dtm)

topic_results.shape

dataset['Topic'] = topic_results.argmax(axis=1)

dataset.to_csv('dataset_topic.csv')