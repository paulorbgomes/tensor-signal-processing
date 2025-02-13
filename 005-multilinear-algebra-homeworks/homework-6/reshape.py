'''
Numpy Reshape Function
'''

import numpy as np

'''
arr = np.array([1,2,3,4,5,6])
print(f"{arr} \n")

# 2D array ...
arr_reshaped = arr.reshape(2,3)
print(f"{arr_reshaped} \n")

# 1D array ...
arr_reshaped = arr_reshaped.reshape(-1)
print(f"{arr_reshaped} \n")

# 3D array ...
arr_reshaped = arr_reshaped.reshape(2,3,1)
print(f"{arr_reshaped}")
'''

# order="C" or order="F" ...
'''
arr = np.array([
                    [1,2],
                    [4,5],
                    [7,8]
               ])
print(arr)
print(arr.shape)
print("")

arr_reshaped = arr.reshape(2,-1,order="C")
print(arr_reshaped)
print(arr_reshaped.shape)
print("")

arr_reshaped = arr.reshape(2,-1,order="F")
print(arr_reshaped)
print(arr_reshaped.shape)
'''

# Example ...
arr = np.array([
                [1,2,3,4,5],
                [6,7,8,9,10]
              ])
print(arr)
print(arr.shape)
print("")

arr_reshaped = arr.reshape(5,-1,order="C")
print(arr_reshaped)
print(arr_reshaped.shape)
print("")

arr_reshaped = arr.reshape(5,-1,order="F")
print(arr_reshaped)
print(arr_reshaped.shape)
print("")






















