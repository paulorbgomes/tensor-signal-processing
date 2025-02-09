'''
Numpy Tensors and Slicing
'''

import numpy as np

# 3D Tensor ...
t = np.array([
    [
        [1,-1,11,-11],
        [2,-2,22,-22]
    ],
    [
        [3,-3,33,-33],
        [4,-4,44,-44]
    ],
    [
        [5,-5,55,-55],
        [6,-6,66,-66]
    ]
])
'''
print(t)
print(t.ndim)   # number of dimensions
print(t.shape)  # size of dimensions
print(t.size)   # number of elements
'''

# Slicing Along First Dimension ...
'''
print(t[0,:,:])
print(t[1,:,:])
print(t[2,:,:])
'''

# Slicing Along Second Dimension ...
'''
print(t[:,0,:])
print(t[:,1,:])
'''

# Slicing Along Third Dimension ...
'''
print(t[:,:,0])
print(t[:,:,1])
print(t[:,:,2])
print(t[:,:,3])
'''







