# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 21:05:49 2022

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

url = "bank-full.csv"
data = pd.read_csv(url)


# Tratamiento de la data (Limpiar y normalizar la data)

data.housing.replace(['yes', 'no'], [0, 1], inplace=True)


data.marital.replace(['married', 'single', 'divorced'], [0, 1, 2], inplace=True)
data.marital.value_counts()

data.y.replace(['yes', 'no'], [0, 1], inplace=True)
data.y.value_counts()

data.education.replace(['primary', 'secondary', 'tertiary', 'unknown'], [0, 1, 2, 3], inplace=True)
data.education.value_counts()

data.default.replace(['no', 'yes'], [0, 1], inplace=True)
data.default.value_counts()

data.loan.replace(['yes', 'no'], [0, 1], inplace=True)
data.loan.value_counts()

data.contact.replace(['unknown', 'cellular', 'telephone'], [0, 1, 2], inplace=True)
data.contact.value_counts()

data.poutcome.replace(['unknown', 'success', 'failure', 'other'], [0, 1, 2, 3], inplace=True)
data.poutcome.value_counts()


data.drop(['balance', 'duration', 'campaign', 'pdays', 'previous', 'job', 'day', 'month'], axis=1, inplace=True)

age_mean = data.age.mean()
data.age.replace(np.nan, age_mean, inplace=True)

ranges = [0, 8, 15, 18, 25, 40, 60, 100]
ranges_names = ['1', '2', '3', '4', '5', '6', '7']
data.age = pd.cut(data.age, ranges, labels=ranges_names)
data.age.value_counts()

data.dropna(axis=0, how='any', inplace=True)


# Partir la data en dos
data_train = data[:22606]
data_test = data[22605:]

x = np.array(data_train.drop(['y'], axis=1))
y = np.array(data_test.y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['y'], axis=1))
y_test_out = np.array(data_test.y)


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



