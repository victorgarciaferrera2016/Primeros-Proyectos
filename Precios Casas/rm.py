# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:01:06 2024

@author: victo
"""

import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



#importamos la data
boston = pd.read_csv('BostonHousing.csv')
print(boston)

#imformacion del df
print(boston.columns)

print(boston.size)
print(boston.info())
print('000')

#Regresion polinomial
#seleccionamos solamanete la columna rm
X = boston['rm']
y = boston['medv']

X_p =(X.values).reshape(-1, 1)
y_p =(y.values).reshape(-1, 1)
print(X)


plt.scatter(X_p,y_p)


#separa conjunto de datos de prueba y entrenamiento
X_train_p, X_test_p, y_train_p , y_test_p = train_test_split(X_p, y_p, test_size = 0.2)

polinomio = PolynomialFeatures(degree = 2)

#Se trasnforman las caracteristica existentes en caracteristicas de mayor grado
X_train_polinomio = polinomio.fit_transform(X_train_p)
X_test_polinomio = polinomio.fit_transform(X_test_p)


#
#Define el algoritmo a utilizar
#
#
pr = linear_model.LinearRegression()

#Entreno el modelo 
pr.fit(X_train_polinomio,y_train_p)

#Realizo la predccion con test
Y_pred_pr = pr.predict(X_test_polinomio)

#Graficamos resiultados
plt.scatter(X_test_p,y_test_p)
plt.scatter(X_test_p,Y_pred_pr,color='red',linewidths=6)
plt.show()

#CAlculando coeficientes
print(pr.coef_)
print(pr.intercept_)


#precision dle algoritmo 
print(pr.score(X_train_polinomio,y_train_p))

