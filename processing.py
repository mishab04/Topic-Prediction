# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 21:53:28 2019

@author: Abhinav
"""

import pandas as pd
#import numpy as np
import re
from nltk.corpus import stopwords

def remove_stopwords(text):     
    stop_word_list = pd.DataFrame(pd.read_csv('stop_words.csv')).values.tolist()
    stop_words_final = stopwords.words("english")
    stop_words_final.extend(stop_word_list)
    filter_word = [word for word in text.split() if word not in stop_words_final]
    filter_text = ' '.join(filter_word)
    return filter_text

def remove_shortwords(text):
    short_word = [word for word in text.split() if len(word) > 3]
    short_text = ' '.join(short_word)
    return short_text

def processing_text(text):
    text = text.lower()
    text = re.sub('[^\w\s]',' ',text)
    text = re.sub('[012345789]',' ',text)
    text = remove_stopwords(text)
    text = remove_shortwords(text)
    return(text)

class file_processing:
    
    def __init__(self,filename = None):
        self.filename = filename
    
    def readdata(self,filename):
        self.filename = filename
        dt = pd.read_csv(filename,encoding = 'latin-1')
        return dt
        

class text_processing:
    
    def __init__(self,dt = None):
        self.dt = dt
             
    def text_process(self,dt):
        dt['Content'] = dt['Content'].apply(lambda x: processing_text(x))
        return dt
        