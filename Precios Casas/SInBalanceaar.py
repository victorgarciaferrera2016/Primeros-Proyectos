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


###########################################333
###### #####################################333
######  #####################################333
### SIN BALANCEAR CHAS #####################################333
### SIN BALANCEAR CHAS #####################################333
### SIN BALANCEAR CHAS #####################################333
### SIN BALANCEAR CHAS #####################################333
### SIN BALANCEAR CHAS #####################################333
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
    
    #se asigna nombre y modelo a name y modelo
    print(f'Procesando Modelo {name}-----------------')
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    
    #Metricas 
    accuracy    =  accuracy_score(y_test, predicciones)
    precision   = precision_score(y_test, predicciones)
    recall      =    recall_score(y_test, predicciones)
    f1          =        f1_score(y_test, predicciones)
   
    
    if hasattr(modelo, "predict_proba"):
            proba = modelo.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, proba[:, 1])
    else:
            roc_auc = None
 
    # Incorporar métricas a la lista de resultados
    results['Model'].append(name)
    results['Accuracy'].append(accuracy)
    results['Precision'].append(precision)
    results['Recall'].append(recall)
    results['F1-score'].append(f1)
    results['AUC-ROC'].append(roc_auc)
 
    
#
#########Graficando resultados.###########
#
#con esto cambio por 0 los none
for metrica in results:
        results[metrica] = [0 if valor is None else valor for valor in results[metrica]]

#
#  GRAFICO BARRAS DE METRICAS SIN AJUSTAR
# Crear gráfico
plt.figure(figsize=(10, 7))

# Anchura de las barras
bar_width = 0.15

# Definir colores para las barras
colores = ['skyblue', 'salmon', 'lightgreen', 'orange', 'pink']

medidas = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC']
valores = [results[columna][fila_color] for columna in results]


ind = np.arange(len(results['Model']))
print(ind)
for i, metrica in enumerate(medidas):
    plt.bar(ind + i * bar_width, 
            results[metrica], 
            bar_width, 
            label=metrica, 
            color=colores[i])

# Agregar valores de métricas debajo del gráfico
for i, modelo in enumerate(results['Model']):
    for j, metrica in enumerate(medidas):
        valor = results[metrica][i]
        plt.text(ind[i] + j * bar_width - 0.03, valor + 0.005, f'{valor:.2f}', fontsize=8)


# Configuraciones adicionales
plt.xlabel('Modelos')
plt.ylabel('Valor')
plt.title('Comparación de Métricas por Modelo con variable Target Sin balanceada')
plt.xticks(ind + bar_width * (len(medidas) - 1) / 2, results['Model'])
plt.legend()
plt.tight_layout()

