#regresion lineal simple
#importar librerias
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
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
#    print(  boston[columna].value_counts().index   )
#    print()


# Crean y ejecutando modelos 

#!pip install sklearn.metrics
from sklearn import metrics

modelos = {
        'Regresión Logística'         : LogisticRegression(),
        'Árbol de Decisión'           : DecisionTreeClassifier(),
        'Random Forest'               : RandomForestClassifier(),
        'Naive Bayes'                 : GaussianNB()
        
        }



X = boston['rm'].values
X = X.reshape(-1, 1)
y = boston['medv'].values


# Implementando un regresion lineal simple

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2)


#selecciionando el modelo o algoritmo
algoritmo_lr = linear_model.LinearRegression()


#entreno el modelo le apso los subconjuntos de datos de entrenamiento
algoritmo_lr.fit(X_train,y_train)


# Pruebo el modelo o Predccion

Y_prediccion = algoritmo_lr.predict(X_test)
print('  Prediccion         Test')
print('------------------------------------')
for i in range(0,10,1):
    print(Y_prediccion[i],"    ", y_test[i])
  

print('Datos del modelo ----')
print(f'coeficiente a o pendiente : {algoritmo_lr.coef_  }')
print(f' Intercepto               : {algoritmo_lr.intercept_  }')
print(f'La ecucion del modelo es : y = {algoritmo_lr.coef_  }x + {algoritmo_lr.intercept_  } ')

#precision del algoritmo 
#mtricas resulantes y analíticas del algoritmo
print(f'Precion del modelo :  {algoritmo_lr.score(X_train,y_train)  }' )
print( ind)











 