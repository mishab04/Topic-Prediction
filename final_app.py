# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 22:47:26 2019

@author: Abhinav
"""

from processing import file_processing

fp = file_processing()

dt = fp.readdata('Disease Classification-Disease_Treatment.csv')

from processing import text_processing

tp = text_processing()

df = tp.text_process(dt.copy())

from prediction import model_prediction

mp = model_prediction()

df = mp.make_prediction(df)