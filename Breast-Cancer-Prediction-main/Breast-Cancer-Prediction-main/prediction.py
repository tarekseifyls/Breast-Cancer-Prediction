# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 13:40:47 2023

@author: Acer
"""

import joblib

def predict(temp):
    model = joblib.load('best_model_breast.sav')
    return model.predict(temp)