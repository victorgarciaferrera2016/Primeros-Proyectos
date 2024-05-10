# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 20:18:03 2024

@author: victo
"""

import numpy as np


mat1 = np.array( [[2,2,2], [2,2,4] , [2,2,8] ])
print(mat1)
type(mat1)

mat2 = [         [11,22,33] , ['aa','ab','ax'], ['f',1,2]      ]
mat2
type(mat2)

mat3 = np.array([         [11,22,33] , ['aa','ab','ax'], ['f',1,2]      ]).reshape(3,3)
type(mat3)

type(mat3.shape)

mat = np.arange(0,9).reshape(3,3)
mat[1:3,0:2]
mat1
np.sum(mat1)
mat1[0].sum()
mat1
np.sum(mat1,axis=0)
dic = dict([(1,'agua'),(2,'luz')])
dic.items()
for i , j in dic.items():
    print(i,"   ",j )
    print(dic.values())
