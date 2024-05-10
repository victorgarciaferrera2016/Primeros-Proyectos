# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 01:40:23 2024

@author: victo
"""

#Import Librerias


#lote de librerias para majejo de de estructuras de datos y visualización.
import numpy              as np
import pandas             as pd
import matplotlib.pyplot  as plt
import seaborn            as sns
import random 

#Lote de Librerias para Preparacion de Datos.

#Lote de Librerias para Preparacion de Datos.

from sklearn.preprocessing  import OneHotEncoder, OrdinalEncoder
from sklearn.impute         import SimpleImputer
from sklearn.compose        import  make_column_transformer
from sklearn.model_selection import train_test_split


#Lote de Librerias para Modelar .

from sklearn.linear_model import LinearRegression
from sklearn              import metrics


#configurando la pantalla praa despliegue de datos.

pd.set_option('display.max_rows',90)
pd.set_option('display.max_columns',90)

#importar datos

df = pd.read_csv('F:/CAPACITACION/MODELO MLEARNING/CURSO DR ARRIGO/Recursos/house-prices-advanced-regression-techniques/train.csv')
#df.isnull().sum()
#df.columns
#df.info()


miembros = {'Victor':100 , 'Paola':100}

#Generadno las semillas para numpy y para random
np.random.seed(sum(miembros.values()))
random.seed(sum(miembros.values()))

#seleccionando las columnas para el proyecto. Seleeciono 20 columnas.


#comando original
#columnas_para_proyecto = random.choices ( list(df[1:80]) , k =20)
columnas_para_proyecto = random.sample( list(df[1:80]) , k =20)

df = df[columnas_para_proyecto + ['SalePrice']]


# VErifico si el df a analizar tiene columnas duplicadas
# Verificar si hay columnas duplicadas
columnas_duplicadas = df.columns[df.columns.duplicated()]

if columnas_duplicadas.any():
    print("El DataFrame tiene columnas duplicadas:")
    print(columnas_duplicadas)
else:
    print("El DataFrame no tiene columnas duplicadas.")


#############################################################################
#y que el % de nulos sean <= 20%

# Calcula el porcentaje de valores nulos para cada columna. Genera una serie
# Filtra las columnas que tienen un porcentaje de nulos menor al 20%

minimo = 0.2
porcentaje_nulos = df.isnull().mean()   #serie
columnas_para_eliminar = porcentaje_nulos[porcentaje_nulos > minimo ].index.tolist() #lista

promedio = df['GarageType'].isnull().mean()

#Genero un df con nulos, % nulos y accion a tomar


df_nulos      = df.isnull().sum()    # genera una serie
df_types      = df.dtypes            # genera una serie
df_Por_Nulos  = df.isnull().mean()



df_info = pd.DataFrame({'Info':df_types, 'Nulos':df_nulos, '%Nulos':df_Por_Nulos})

#agrego columna accion a tomar por cada tipo de dato
df_info['Accion'] = df_info.apply(
                             lambda row: 
                             'Eliminar'  if row['%Nulos'] > minimo  
                             else 'Nada' if row['%Nulos']  == 0
                             else 'Moda' if row['Info']    == 'object' 
                             else 'Media',
                             axis=1)

#Agrego el dato a incluir en la preparacion si asi se requiere.
df_info['Dato'] = df_info.apply(
                             lambda row:  
                             df[row.name].mode()[0]   if ( row['Info'] == 'object' and row['Nulos'] > 0 ) 
                             else df[row.name].mean() if ( row['Info'] != 'object' and row['Nulos'] > 0) 
                             else None,
                             axis = 1 )

#3.- Preparando Datos 
#Borro las columnas marcadas como a eliminar

columnas_a_eliminar = df_info[df_info['Accion'] == 'Eliminar'].index.to_list()
#df                  = df.drop(columns=columnas_a_eliminar)

#itero por df df_info y aplico la acccion a la columna indicada con e ldato
#antes
df.info()

for ind, accion in df_info[df_info['Accion']!= 'Nada']['Accion'].items():
       print(ind,accion)
       if accion == 'Eliminar' :
               df      = df.drop(columns=ind)
       elif accion == 'Moda':
               df[ind] = df[ind].fillna(df[ind].mode()[0])
       elif accion == 'Media':
               df[ind] = df[ind].fillna(df[ind].mean())

    
df.isnull().sum()


# Transformar las variables catagoricas a valores numericos.
#
#  VISUALIZANDO LOS DATOS 
#
# Ajustar espacio entre subgráficos
plt.figure(figsize=(30, 20))
plt.subplots_adjust(hspace=0.5)  # Ajustar espacio vertical subgráficos

variables = list(df.columns)
grafico = 1
for variable in variables:
    
 
    #displot de seeaborn
    print(df[variable].dtype, "         ", variable)
    print(df[variable].value_counts(dropna=False))
    
    if df[variable].dtype != 'object':
        sns.displot(df[variable])
        plt.title('Distribucion :'+ variable)
        plt.xlabel(variable+' Numero',fontsize = 16)
        plt.ylabel('Frecuencia',fontsize = 16)
    else :
      #counplot de seeaborn

        sns.catplot(x = variable, kind='count' , data=df )
        plt.title('Distribucion :'+ variable)
        plt.xlabel(variable + ' Object',fontsize = 16)
        plt.ylabel('Frecuencia',fontsize = 16)
    grafico = grafico +1 
    if grafico == 3: grafico = 1
   
    plt.show()

df.info()

#ANalisisi por Columna para transformar
# Nota 
#1.- Grafircar las variables una por una
#2.- Variables Categoricas-Ordenadas, OrdinalEncoder
#3.- Variables Categoricas No Ordenadas, pd.get_dummies
#4.- sns.paitplot sns.heatmap
#


#0   Street        1460 non-null   object 
# demasiado disribuidad en dos variables
#one Hot encoder

dum_Street = pd.get_dummies(df.Street, 
                            prefix = 'Street' , 
                            drop_first =True,
                            dtype='int')

#1   MasVnrArea    1460 non-null   float64  NADA
#2   LotFrontage   1460 non-null   float64  NADA
#3   OverallCond   1460 non-null   int64   NADA  
#4   BsmtFinSF1    1460 non-null   int64   NADA 
#5   MSSubClass    1460 non-null   int64  NADA  
#6   Fireplaces    1460 non-null   int64    NADA
#7   RoofStyle     1460 non-null   object  
dum_RoofStyle = pd.get_dummies(df.RoofStyle, 
                            prefix = 'RoofStyle' , 
                            drop_first =True,
                            dtype='int')

#8   MSZoning      1460 non-null   object 
dum_MSZoning = pd.get_dummies(df.MSZoning, 
                            prefix = 'MSZoning' , 
                            drop_first =True,
                            dtype='int')
#9   BsmtFinType2  1460 non-null   object Rating Ordinal encore
oe = OrdinalEncoder(categories=[['GLQ','ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']])
dum_BsmtFinType2 = pd.DataFrame(oe.fit_transform(df[['BsmtFinType2']] ))
dum_BsmtFinType2.columns = ['BsmtFinType2_numeric']
dum_BsmtFinType2.head()

#10  PoolArea      1460 non-null   int64  NADA

#11  GarageType    1460 non-null   object No es Rating .Es dumy 
dum_GarageType = pd.get_dummies(df.GarageType, 
                            prefix = 'GarageType' , 
                            drop_first =True,
                            dtype='int')

#12  Foundation    1460 non-null   object  Dummy
dum_Foundation = pd.get_dummies(df.Foundation, 
                            prefix = 'Foundation' , 
                            drop_first =True,
                            dtype='int')

#13  Exterior2nd   1460 non-null   object Hacer Ordinal encouder o Maping
dic_Exterior2nd = {
                'VinylSd'    : 1 ,   'MetalSd'    : 2 , 
                'HdBoard'    : 3 ,   'Wd Sdng'    : 4 , 
                'Plywood'    : 5 ,   'CmentBd'    : 6 , 
                'Wd Shng'    : 7 ,   'Stucco'     : 8 , 
                'BrkFace'    : 9 ,   'AsbShng'    : 10 , 
                'ImStucc'    : 11 ,  'Brk Cmn'    : 12 ,
                'AsphShn'    : 14 ,  'Other'      : 15 , 
                'CBlock'     : 16 ,  'Stone'      : 13 }
df['Exterior2nd'] = df['Exterior2nd'].map(dic_Exterior2nd)
df.Exterior2nd.value_counts()

#14  BldgType      1460 non-null   object object Hacer Ordinal encouder o Maping
dic_BldgType = 	{	
       '1Fam'	:1 ,	'2fmCon'	: 2 , 'Duplex'	: 3 ,
       'TwnhsE'	: 4 , 'Twnhs'	: 5 }
df.BldgType.value_counts()
df['BldgType'] = df['BldgType'].map(dic_BldgType)
df.BldgType.value_counts()

#15  TotalBsmtSF   1460 non-null   int64  NADA
df.TotalBsmtSF.value_counts()
#16  FullBath      1460 non-null   int64  NAdA
df.FullBath.value_counts()
#17  RoofMatl      1460 non-null   object    dumies no rating
dum_RoofMatl = pd.get_dummies(df.RoofMatl, 
                            prefix = 'RoofMatlon' , 
                            drop_first =True,
                            dtype='int')

df.RoofMatl.value_counts()
#18  CentralAir    1460 non-null   object diccionario : 
dic_CentralAir = {
            'Y': 1 , 'N' : 0}
df.CentralAir.value_counts()
df['CentralAir'] = df['CentralAir'].map(dic_CentralAir)
df.CentralAir.value_counts()
#19  SalePrice     1460 non-null   int64  NADA

# Generando los datos final concatenando todo

data = pd.concat([
                dum_Street ,
                df.MasVnrArea,
                df.LotFrontage,
                df.OverallCond, 
                df.BsmtFinSF1,
                df.MSSubClass,
                df.Fireplaces,
                dum_RoofStyle, 
                dum_MSZoning,
                dum_BsmtFinType2,
                df.PoolArea ,
                dum_GarageType ,
                dum_Foundation,
                df.Exterior2nd,
                df.BldgType,
                df.TotalBsmtSF,
                df.FullBath,
                dum_RoofMatl ,
                df.CentralAir,
                df.SalePrice ], axis = 1)

data.describe()
data.info()
data.columns
#sns.pairplot(data)
#sns.heatmap(data.corr(), annot = True)


#genera graficos de forma intantanea pra entendimiento de los datos.
#from pandas.plotting import scatter_matrix
#attributes = data.columns
#scatter_matrix(data[attributes],figsize = (12,8))

X = data.drop(columns=['SalePrice'])
y = data['SalePrice']
#
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=42)

#
# Construyendo un modelo ML
#
lm = LinearRegression()
lm.fit(X_train,y_train)

#Coeficientes
coeff_data = pd.DataFrame(lm.coef_,X.columns,columns=['Coeficientes'])
print(coeff_data)

#
#  PREDICCIONES 
#
predicciones = lm.predict(X_test)

#
# EVALUACION
#
plt.scatter(y_test,predicciones) 

sns.distplot((y_test-predicciones),bins= 50)
plt.hist(y_test-predicciones, bins = 50)

print('MAE :', metrics.mean_absolute_error(y_test,predicciones))
print('MSE :', metrics.mean_squared_error(y_test,predicciones))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test,predicciones)))

true_value = y_test
prediccion_value = predicciones

plt.figure(figsize=(10,10))
plt.scatter(true_value, prediccion_value, c= 'crimson')

p1 = max(max(prediccion_value), max(true_value))
p2 = min(min(prediccion_value), min(true_value))

plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values ', fontsize=15)
plt.ylabel('Predicciones ', fontsize=15)
plt.axis ('equal')
plt.show()

for fila in list(data.columns):
    for columna in list(data.columns):
     
       coef_correlacion = df['x'].corr(df['y'])
data.describe()


