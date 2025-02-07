'''
Homework 4: Least Squares Kronecker Product Factorization (LSKronF)
'''

import numpy as np
import functions as mf
import matplotlib.pyplot as plt

if __name__ == "__main__":
    Ao = np.random.normal(0,1,(3,3))
    a = mf.vec(Ao)
    #print(a)
    Arec = mf.unvec(a,3,3)
    print(mf.nmse(Ao,Arec))

    


    

