# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 08:11:49 2019

@author: Abhinav
"""
import pickle

class model_prediction:
    
    def __init__(self,dt = None):
        self.dt = dt
    
    def make_prediction(self,dt):
        vectorizer = pickle.load(open("vector.pickel", "rb"))
        LDA_model = pickle.load(open("model.pickel", "rb"))
        dtm = vectorizer.transform(dt['Content'])
        topic_results = LDA_model.transform(dtm)
        dt['Topic'] = topic_results.argmax(axis=1)
        return dt