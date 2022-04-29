# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 22:51:31 2022

@author: Luis Torres
"""


import numpy as np
import pandas as pd

# Obtener la data

url = "weatherAUS.csv"
data = pd.read_csv(url)[:3000]

# Tratamiento de la data (Limpiar y normalizar la data)

data.RainToday.replace(['yes', 'no'], [1, 0], inplace=True)
data.RainToday.value_counts()

data.RainTomorrow.replace(['no', 'yes'], [0, 1], inplace=True)
data.RainTomorrow.value_counts()

data.drop(['Date', 'Location', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm'], axis=1, inplace=True)

data.dropna(axis=0, how='any', inplace=True)



