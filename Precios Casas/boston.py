#regresion lineal simple
#importar librerias
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.datasets import load_boston, load_diabetes, load_digits
import matplotlib as plt

boston = datasets.boston()


print('Informacion en el Dataser :')
print(boston.keys())

print()