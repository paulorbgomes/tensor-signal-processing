'''
Introducao ao Matplotlib
https://matplotlib.org
'''

import numpy as np
import matplotlib.pyplot as plt

# Dataset ...
'''
numbers = [1, 2, 3, 4, 5]
squares = [1, 4, 9, 16, 25]
'''

input_values = np.array([1, 2, 3, 4, 5])
squares = np.array(input_values ** 2)

plt.style.use('fast')

fig, ax = plt.subplots()   # subplots() permite gerar varios graficos na mesma figura ...
ax.plot(input_values, input_values, linewidth=3)
# Define o titulo do grafico e os eixos do rotulo
ax.set_title("Number Analysis", fontsize=14)
ax.set_xlabel("Value", fontsize=12)
ax.set_ylabel("Result", fontsize=12)
# Define o tamanho dos rotulos de marcacao de escala
ax.tick_params(labelsize=14)

ax.plot(input_values, squares, linewidth=3)

plt.show()   # abre o visualizador da matplotlib ...
