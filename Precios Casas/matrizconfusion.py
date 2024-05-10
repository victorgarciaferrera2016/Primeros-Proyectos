# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 01:52:03 2024

@author: victo
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:01:44 2024

@author: victo
"""
#regresion lineal simple
#importar librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
from sklearn import datasets, linear_model , metrics
from sklearn.model_selection import train_test_split #divide bases en datos de entrenamiento (aprender) y prueba (pronosticar)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score,recall_score, f1_score #Metricas de evaluacion de los modelos utilizados
from sklearn.model_selection import train_test_split

boston = pd.read_csv('BostonHousing.csv')
pd.set_option('display.max_columns',None)


#definicion del lienzo
plt.figure(figsize = (8,6))

#grafico de barras
eje_x=boston['rad'].value_counts().sort_index()


y = boston['chas']
X = boston.drop(columns='chas')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2 , random_state=77 )

modelos = {
        'Regresión Logística'         : LogisticRegression(),
        'Árbol de Decisión'           : DecisionTreeClassifier(),
        'Random Forest'               : RandomForestClassifier(),
        'Naive Bayes'                 : GaussianNB()
         }

# en este dic/lista , acumulo los resultados
results = { 'Model'    : [], 
               'Accuracy' : [], 
               'Precision': [], 
               'Recall'   : [], 
               'F1-score' : [], 
               'AUC-ROC'  : []}

fila_color = 0
colores = ['skyblue', 'salmon', 'lightgreen', 'orange', 'pink']
for name, modelo in modelos.items():
    plt.figure(figsize = (8,6))
    #se asigna nombre y modelo a name y modelo
    print(f'Procesando Modelo {name}-----------------')
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    
    #
    # Luego de generar el modelo, comparo resultdo modelo, con predecciones 
    # genero matriz de confusion
    #
    ### grafico matriz de confusuin
    conf_matrix = metrics.confusion_matrix(y_test, predicciones)
    sns.heatmap(conf_matrix, 
               annot=True, 
                fmt='d', 
               cmap='Purples', 
                xticklabels=['No default', 'Default'], 
                yticklabels=['No default', 'Default']
                )
    plt.title(f'Matrix de Confusión {name}', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
       
    for i in (0,1):
        for j in (0,1):
           plt.text(j + 0.5, i + 0.5, f'{conf_matrix[i,j]:.2f}',
                           ha="center", va="center", color="black", fontsize=20)
    plt.show() 

    
    