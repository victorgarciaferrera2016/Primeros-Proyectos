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
print(boston.columns)

#valindando la existencia de objetos
print(boston.info())

#saca ls valores la correncia por cada valor en la columna
# tiene sentido para variables/columnas objets
print(  boston['rm'].value_counts().index   )

# no tiene sentido, pues se debe hacer sobre variables objets
# para poder normalizar o llevar a numeros o dumies.
#for columna in boston.columns:
#    print("Columna : " , columna , " Diferentes : ",len(boston[columna].value_counts().index) )
#    print(boston[columna].value_counts()     )
#    print()

#definicion del lienzo
plt.figure(figsize = (40,20))

#grafico de barras
eje_x=boston['rad'].value_counts().sort_index()

# posicionando el grafico
#   1 X
#   X X 
plt.subplot(2, 2, 1)
ax = sns.countplot(x='rad', data = boston)
# con este , manejo tamaño etiquetas eje x 
ax.set_xticklabels(ax.get_xticklabels(), fontsize=35)

# Añadir etiquetas al tope de cada barra por dentro
for i, valor in enumerate(eje_x):
    plt.text(i, valor-5, str(valor), ha='center', va='bottom', fontsize=25)


#grafico de torta
# posicionando el grafico
#   X 2
#   X X 
plt.subplot(2, 2, 2)
ax=plt.pie(boston['rad'].value_counts().sort_index(),labels=eje_x, 
           autopct='%1.1f%%', # con esto pongo las etiquetas dentro 
           textprops={'color': 'black', 'fontsize': 30}) # formato de la etiquetas
plt.ylabel('Cantidad')


#grafico de torta
# posicionando el grafico
#   X X
#   3 X 
plt.subplot(2, 2, 3)
ax = sns.countplot(x='chas', data = boston)
# con este , manejo tamaño etiquetas eje x 
ax.set_xticklabels(ax.get_xticklabels(), fontsize=35)

#ordeno al eje x para que quede igual que el orden de las barras
eje_x=boston['chas'].value_counts().sort_index()
# Añadir etiquetas al tope de cada barra por dentro
for i, valor in enumerate(eje_x):
    plt.text(i, valor-40, str(valor), ha='center', va='bottom', fontsize=25)


########################BOXPLOT####################################
#Otro Lienzo para graficos tipo boxplot
plt.figure(figsize = (30,20))

#Graficos de BoxPlot para ver outliers

#variables_discretas = ['rad','chas']
variables_discretas  = list(boston.columns)
variable_target      = 'medv'
posicion = 1
for grafico in variables_discretas:
    
    plt.subplot(7, 2, posicion)
    sns.boxplot( x=grafico, y=variable_target, data = boston)
    
    plt.title(grafico.upper(), fontsize=25)  
    plt.xlabel(grafico.upper(), fontsize=25)
    plt.ylabel(variable_target.upper(), fontsize=25)
    
    posicion = posicion + 1
    #genera un lienzo cada 4 graficos
    #if posicion == 5:
    #           posicion = 1
     #          plt.figure(figsize = (30,20))

plt.tight_layout()

###################VIOLIN#################################3
#Otro Lienzo para graficos tipo violin
plt.figure(figsize = (30,20))

#Graficos de BoxPlot para ver outliers

#variables_discretas = ['rad','chas']
variables_discretas  = list(boston.columns)
variable_target      = 'medv'
posicion = 1
for grafico in variables_discretas:
    
    plt.subplot(7, 2, posicion)
    sns.violinplot( x=grafico, y=variable_target, data = boston)
    
    plt.title(grafico.upper(), fontsize=25)  
    plt.xlabel(grafico.upper()        , fontsize=25)
    plt.ylabel(variable_target.upper(), fontsize=25)
    
    
    posicion = posicion + 1
    #genera un lienzo cada 4 graficos
    #if posicion == 5:
    #           posicion = 1
     #          plt.figure(figsize = (30,20))

plt.tight_layout()


##################HISTOGRAMAS###################################
#Otro Lienzo para graficos tipo violin
plt.figure(figsize = (30,20))

#Graficos de BoxPlot para ver outliers

#variables_discretas = ['rad','chas']
variables_discretas  = list(boston.columns)
variable_target    = 'medv'
posicion = 1
for grafico in variables_discretas:
    
    plt.subplot(7, 2, posicion)
    sns.histplot(data=boston,x=grafico)
    
    plt.title(grafico.upper(), fontsize=25)
    plt.xlabel(grafico.upper(), fontsize=25)
    plt.ylabel('FRECUENCIA', fontsize=20)
    
    
    posicion = posicion + 1
    #genera un lienzo cada 4 graficos
    #if posicion == 5:
    #           posicion = 1
     #          plt.figure(figsize = (30,20))

plt.tight_layout()


##################DISPERSION###################################
#Otro Lienzo para graficos tipo scatter
plt.figure(figsize = (30,20))

variables_discretas  = list(boston.columns)
variable_target    = 'medv'
posicion = 1
for grafico in variables_discretas:
    if grafico not in ('rad','chas','medv'):
    
        plt.subplot(7, 2, posicion)
        sns.scatterplot(data=boston,x=grafico, y = variable_target, s=40)

        #Etiquetas del grafico
    
        plt.title(grafico.upper(), fontsize=25)
        plt.xlabel(grafico.upper(), fontsize=25)
        plt.ylabel(variable_target.upper(), fontsize=25)
       
        # Agregar línea de regresión
        sns.regplot(data=boston, x=grafico, y='medv', scatter=False,color='red')
           
        posicion = posicion + 1
        #if posicion == 5:
        #            posicion = 1
        #            plt.figure(figsize = (30,20))
   
# Ajustar el diseño para evitar superposiciones
plt.tight_layout()
plt.show()


# Crear el mapa de calor de matriz de correlacion
plt.figure(figsize=(30,10))  # Tamaño de la figura
matriz_correlacion = boston.corr()
heatmap = sns.heatmap(matriz_correlacion, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f", 
            xticklabels=True, 
            yticklabels=True,
            linewidths = 1,
            linecolor = 'black' ) # 'coolwarm' para la paleta de colores

plt.title("Mapa de calor de correlación entre variables", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=15)

heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90, verticalalignment='top')
# Agranda los valores de etiqueta de cada cuadrado

for i in range(len(matriz_correlacion)):
    for j in range(len(matriz_correlacion)):
       heatmap.text(j + 0.5, i + 0.5, f'{matriz_correlacion.iloc[i,j]:.2f}',
                       ha="center", va="center", color="black", fontsize=20)
plt.tight_layout()
plt.show()

#
#analizando modelos de residuos  
#
import statsmodels.api as sm

# Ajustar el modelo de regresión lineal
modelo1 = sm.OLS(boston['medv'], sm.add_constant(boston['lstat'])).fit()

# Mostrar un resumen del modelo
print(modelo1.summary())

#print(results.keys())

#
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
   
    #
    # Luego de generar el modelo, comparo resultdo modelo, con predecciones 
    # genero matriz de confusion
    #
    #conf_matrix = metrics.confusion_matrix(y_test, predicciones)
    #sns.heatmap(conf_matrix, 
    #            annot=True, 
    #            fmt='d', 
    #            cmap='Purples', 
    #            xticklabels=['No default', 'Default'], 
    #            yticklabels=['No default', 'Default']
    #            )
    #### grafico matriz de confusuin
    #plt.title(f'Matrix de Confusión {name}', fontsize=20)
    #plt.xticks(fontsize=15)
    #plt.yticks(fontsize=15)
    #for i in (0,1):
    #    for j in (0,1):
    #       plt.text(j + 0.5, i + 0.5, f'{conf_matrix[i,j]:.2f}',
    #                       ha="center", va="center", color="black", fontsize=20)
    #plt.show() 
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
   
 






