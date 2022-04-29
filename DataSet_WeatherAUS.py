# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 22:51:31 2022

@author: Luis Torres
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Obtener la data

url = "weatherAUS.csv"
data = pd.read_csv(url)[:3000]

# Tratamiento de la data (Limpiar y normalizar la data)

data.RainToday.replace(['Yes', 'No'], [1, 0], inplace=True)
data.RainToday.value_counts()

data.RainTomorrow.replace(['No', 'Yes'], [0, 1], inplace=True)
data.RainTomorrow.value_counts()

data.drop(['Date', 'Location', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm'], axis=1, inplace=True)

data.dropna(axis=0, how='any', inplace=True)


# Partir la data en dos
data_train = data[:471]
data_test = data[471:]

x = np.array(data_train.drop(['RainTomorrow'], axis=1))
y = np.array(data_test.RainTomorrow)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['RainTomorrow'], axis=1))
y_test_out = np.array(data_test.RainTomorrow)


# Regresion logistica

# Seleccionar un modelo

logreg = LogisticRegression(solver='lbfgs', max_iter=7600)

# Entrenar el modelo

logreg.fit(x_train, y_train)

# Metricas

print('*'*50)
print('Regresion Logistica')

# Accuracy de Test de Entrenamiento

print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de entrenamiento de Entrenamiento

print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de validacion

print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test_out, y_test_out)}')


# Maquina de soporte vectorial

# Seleccionar un modelo

svc = SVC(gamma='auto')

# Entrenar el modelo

svc.fit(x_train, y_train)

# Metricas

print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Test de Entrenamiento

print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de entrenamiento de Entrenamiento

print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de validacion

print(f'accuracy de Test de Entrenamiento: {svc.score(x_test_out, y_test_out)}')


# Arbol de decisión

# Seleccionar un modelo

tree = DecisionTreeClassifier()

# Entrenar el modelo

tree.fit(x_train, y_train)

# Metricas

print('*'*50)
print('Arbol de decisión')

# Accuracy de Test de Entrenamiento

print(f'accuracy de Test de Entrenamiento: {tree.score(x_test, y_test)}')

# Accuracy de entrenamiento de Entrenamiento

print(f'accuracy de Entrenamiento de Entrenamiento: {tree.score(x_train, y_train)}')

# Accuracy de validacion

print(f'accuracy de Test de Entrenamiento: {tree.score(x_test_out, y_test_out)}')


# K-Nearest neighbors

# Seleccionar un modelo

kneighbors = KNeighborsClassifier()

# Entrenar el modelo

kneighbors.fit(x_train, y_train)

# Metricas

print('*'*50)
print('K-Nearest neighbors')

# Accuracy de Test de Entrenamiento

print(f'accuracy de Test de Entrenamiento: {kneighbors.score(x_test, y_test)}')

# Accuracy de entrenamiento de Entrenamiento

print(f'accuracy de Entrenamiento de Entrenamiento: {kneighbors.score(x_train, y_train)}')

# Accuracy de validacion

print(f'accuracy de Test de Entrenamiento: {kneighbors.score(x_test_out, y_test_out)}')


# Random Forest

# Seleccionar un modelo

random_forest = RandomForestClassifier()

# Entrenar el modelo

random_forest.fit(x_train, y_train)

# Metricas

print('*'*50)
print('Random Forest')

# Accuracy de Test de Entrenamiento

print(f'accuracy de Test de Entrenamiento: {random_forest.score(x_test, y_test)}')

# Accuracy de Entrenamiento de Entrenamiento

print(f'accuracy de Entrenamiento de Entrenamiento: {random_forest.score(x_train, y_train)}')

# Accuracy de validacion

print(f'accuracy de Test de Entrenamiento: {random_forest.score(x_test_out, y_test_out)}')




