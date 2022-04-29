# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 22:03:44 2022

@author: Luis Torres
"""

import numpy as np
import pandas as pd

# Obtener la data

url = "diabetes.csv"
data = pd.read_csv(url)


# Tratamiento de la data (Limpiar y normalizar la data)

data.drop(['Pregnancies', 'BloodPressure', 'SkinThickness'], axis=1, inplace=True)


ranges = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ranges_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
data.Age = pd.cut(data.Age, ranges, labels=ranges_names)
data.Age.value_counts()

data.dropna(axis=0, how='any', inplace=True)

