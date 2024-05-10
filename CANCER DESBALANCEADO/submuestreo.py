# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 23:39:23 2024

@author: victo
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

def matriz_de_confusion(clases_reales, clases_predichas,titulo):
    
    print('Reales : '  , clases_reales.values.reshape(-1) )
    print('Predichas :', clases_predichas)
    matriz = confusion_matrix(clases_reales, clases_predichas)
    accuracy = accuracy_score(clases_reales, clases_predichas)


    plt.figure(figsize=(4,4))
    matriz = pd.DataFrame(matriz, columns=['0 : Sana','1 : Cancer'])
    plt.matshow(matriz,cmap='Pastel1',vmin=0, vmax = 20,fignum=1)
    plt.xticks(range(len(matriz.columns)), matriz.columns, rotation = 45)
    plt.yticks(range(len(matriz.columns)), matriz.columns)
    etiquetas = (('Verdaderos\nnegativos', 'Falsos\npositivos'))
    
    plt.text(1.60,-0.30, titulo, fontsize=25,c='red')
    plt.text(2.1, 0.10, 'Accuracy : %0.2f ' % accuracy )    
    
    for i in range(len(matriz.columns)):
        for j in range(len(matriz.columns)):
            plt.text(i, j + 0.14, str(matriz.iloc[i,j]) , fontsize = 30   , ha = 'center', va= 'center')
            plt.text(i, j + 0.25, etiquetas[i][j]  , fontsize = 11.5 , ha = 'center', va= 'center')
    plt.show()
    
    
    #
    #   DATOS CANCER MAMA
    #   0 = SANAS     1 = CANCER
    #
    #    CLASES DESBALANCEADAS
    
    personas = pd.read_csv('F:/CAPACITACION/MODELO MLEARNING/CANCER DESBALANCEADO/cancer_desbalance.csv',header=None)
    prueba = pd.read_csv('F:/CAPACITACION/MODELO MLEARNING/CANCER DESBALANCEADO/cancer_prueba.csv',header=None)
    
    personas.shape
    num_sanas = personas[personas[30]==0][1].size
    num_cancer = personas[personas[30]==1][1].size
    
    plt.bar(['Sansa (%d)' % num_sanas, 'Cancer (%d) ' % num_cancer],
            [num_sanas,num_cancer],
            color = ['cyan','red'],
            width = 0.8
            )
    plt.ylabel('Personas')
    plt.show()
    
    personas_sanas = personas[personas[30]==0]
    personas_cancer = personas[personas[30]==1]
    
    #
    #   SOBREMUESTREO
    #
    sobremuestreo_cancer = personas_cancer.sample(n=290, replace=True, random_state=0)
    submuestreo_sanas = personas_sanas.sample(n=20, replace=False, random_state=0)
    
    sobremuestreo = pd.concat([sobremuestreo_cancer , personas_sanas])
    x_sobremuestreo = sobremuestreo.iloc[ : , :-1 ]
    y_sobremuestreo = sobremuestreo.iloc[ : , -1: ]
    
    submuestreo   = pd.concat([submuestreo_sanas    , personas_cancer])
    x_submuestreo = submuestreo.iloc[ : , :-1 ]
    y_submuestreo = submuestreo.iloc[ : , -1: ]
    
    desbalanceado = pd.concat([personas_sanas,personas_cancer])
    x_desbalanceado = desbalanceado.iloc[ : , :-1 ]
    y_desbalanceado = desbalanceado.iloc[ : , -1: ]
    
    #
    #   Datos de Prueba 
    #
    x_prueba = prueba.iloc[:, :-1]
    y_prueba = prueba.iloc[:, -1:]
    
    #
    #  Modelo Regresión Logística
    #
    
    # DESBALANCEADO
    modelo = LogisticRegression().fit(x_desbalanceado.values,y_desbalanceado.values.reshape(-1) )
    y_pred_desbalanceado = modelo.predict(x_prueba)
    matriz_de_confusion(y_prueba , y_pred_desbalanceado,'Desbalanceado')
    
    #SUBMUESTREO
    modelo = LogisticRegression().fit(x_submuestreo.values,y_submuestreo.values.reshape(-1) )
    y_pred_submuestreo = modelo.predict(x_prueba)
    matriz_de_confusion(y_prueba , y_pred_submuestreo,'Submuestreo')
     
    #SOBREMUESTREO
    modelo = LogisticRegression().fit(x_sobremuestreo.values,y_sobremuestreo.values.reshape(-1) )
    y_pred_sobremuestreo = modelo.predict(x_prueba)
    matriz_de_confusion(y_prueba , y_pred_sobremuestreo,'Sobremuestreo')
    
    
    
    
    
    
    