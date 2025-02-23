import numpy as np
import matplotlib.pyplot as plt

y = np.random.normal(10, 5, (50))
x = np.arange(1,51)

fig, ax = plt.subplots()
ax.plot(x,y,linewidth=2)
ax.set_title("Observações", fontsize=14)
ax.set_xlabel("Amostra", fontsize=12)
ax.set_ylabel("Valor", fontsize=12)

plt.show()
