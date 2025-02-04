import numpy as np
import matplotlib.pyplot as plt

# Dataset ...
x = np.array([1, 2, 3, 4, 5])
y = np.array(x ** 2)

fig, ax = plt.subplots()
ax.scatter(x,y,s=100)
ax.set_title("Título ...", fontsize=14)
ax.set_xlabel("Eixo-x ...", fontsize=12)
ax.set_ylabel("Eixo-y ...", fontsize=12)
ax.axis([0, 10, 0, 50])

plt.show()
