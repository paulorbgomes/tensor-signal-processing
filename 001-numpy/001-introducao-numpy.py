'''
Introducao ao Numerical Python (Numpy)
Link: http://www.opl.ufc.br/post/numpy/
'''

import numpy as np

# Criacao do ndarray:
v = np.array([1, 2, 3, 4])
print(v)

print(v.dtype)   # tipo dos elementos do ndarray
v = np.array([1, 2, 3, 4], dtype="float64")
print(v.dtype)

v = np.array([1.0, 2.0, 3.0, 4.0])
print(v.dtype)

# Indicacao da forma do ndarray:
print(v.shape)

# Funcao reshape():
v = np.array([1, 2, 3, 4]).reshape(2, 2)
print(v)
print(v.shape)

# Numero de eixos ou dimensoes do ndarray:
v = np.array(range(50)).reshape(2, 5, 5)
print(f"Shape = {v.shape}")
print(f"Numero de dimensoes = {v.ndim}")
print(f"Numero de elementos = {v.size}")

# Funcoes zero, ones e diag:
V = np.zeros((3,3))
print(V)
U = np.ones((3, 3))
print(U)
D = np.diag([10, 10, 10])
print(D)

# Funcoes arange e linspace:
v = np.arange(0, 5, 0.5)
print(v)
u = np.linspace(0, 5, 10)
print(u)

# Metodo flatten - retorna uma copia do nparray com todos os elementos em apenas uma dimensao:
v = np.array(range(50)).reshape(2, 5, 5)
print(v.flatten())

# Atributo flat - retorna um iterador nos elementos de um nparray:
v = np.array(range(50)).reshape(2, 5, 5)
v_iter = v.flat
for i in v_iter:
    print(i, end=' ')
print("")

# Indexacao de nparrays:
v = np.array([10, 20, 30, 40]).reshape(2, 2)
print(v)
print(v[0, 1])

v = np.arange(8).reshape(2, 2, 2)
print(v)
print(v[0, 0, 1])

# Fatiamento:
v = np.arange(10)
print(v)
print(v[1:3])

V = np.arange(15).reshape(3, 5)
print(V)
print(V[0:2,2:4])

# Filtrando as colunas 2 e 4 de uma matriz:
A = np.arange(10).reshape(2,5)
B = A[:,[2,4]]
print(A)
print(B)

# Operacoes sobre nparrays:
v = np.array([10, 20, 30])
u = np.array([2, 2, 2])
w = u + v
print(w)

w = u * v
print(w)

w = u / v
print(w)

x = np.array([10, 20])
y = x ** 2
print(y)

# Funcoes especificas:
x = np.arange(10)
media = x.mean()
menor_valor = x.min()
maior_valor = x.max()
print(f"Media = {media}")
print(f"Menor valor = {menor_valor}")
print(f"Maior valor = {maior_valor}")

# Em nparrays (matrizes e tensores) pode-se identificar o eixo em que se deseja aplicar as funcoes
# min, max, mean, etc. Caso o eixo nao seja especificado, a funcao retornara a resposta entre todos
# os valores do nparray ...
A = np.array([10, 20, 30, 40]).reshape(2, 2)
menor = A.min()
menor_colunas = A.min(axis=0)
print(f"A = {A}")
print(f"Menor valor: {menor}")
print(f"Menor valor em cada coluna: {menor_colunas}")

# Produto interno:
v = np.array([10, 20, 30])
u = np.array([2, 2, 2])
print(np.dot(u,v))

# Multiplicacao de um escalar por um vetor:
x = 10 * w
print(w)
print(x)

# Matrizes:
A = np.array([[10,20], [30,40]])
print(A)

I = np.eye(5)   # criacao de uma matriz identidade
print(I)

D = np.diag(np.arange(5))
print(D)

# Operacoes sobre matrizes:
# 1. Multiplicacao de vetores e matrizes e feita por meio da funcao np.dot() ...
v = np.array([10, 10])
A = np.arange(4).reshape(2, 2)
print(v.shape)   # (2,)
print(A.shape)   # (2,2)
u = np.dot(A,v)
print(u)
print(u.shape)

A = np.ones((2,2))
B = 10 * np.ones((2,2))
C = np.dot(A,B)
print(C)

# 2. Transposicao de matrizes pode ser feita com o metodo .transpose() ou .T
u = np.array([10, 20])
print(u)
print(u.transpose())
print(u.T)

A = np.arange(4).reshape(2,2)
print(f"A = {A}")
print(f"Transposta de A = {A.transpose()}")
print(f"Transposta de A = {A.T}")

# Funcoes universais:
u = np.arange(5)
v = np.exp(u)
print(v)

v = np.sin(u)
print(v)

# Geracao de numeros aleatorios: submodulo especifico np.random
# Permite diretamente a criacao de nparrays com valores aleatorios
# 1. Criacao de um nparray segundo uma distribuicao uniforme no intervalo [0,1)
v = np.random.rand(4,4)
print(v)

# 2. Criacao de um nparray segundo uma distribuicao normal com media = 10 e variancia = 1
v = np.random.normal(10, 1, (4,4))
print(v)

# 3. Criacao de um nparray segundo uma distribuicao normal com media = 0 e variancia = 1
v = np.random.normal(0, 1, (4,4))
print(v)

# Fundamentos de algebra linear: submodulo especifico np.linalg
# Possibilita resolver sistemas lineares, inverter matrizes, calcular autovalores e autovetores ...
# 1. Solucao de sistemas lineares Ax = b
A = np.array([10, 20, 30, 40]).reshape(2,2)
b = np.array([5, 10])
x = np.linalg.solve(A, b)
print(x)

# 2. Rank e determinante
A = np.array([10, 20, 30, 40]).reshape(2,2)
rankA = np.linalg.matrix_rank(A)
print(f"Rank(A) = {rankA}")
detA = np.linalg.det(A)
print(f"Det(A) = {detA}")

# 3. Matriz inversa
A = np.array([10, 20, 30, 40]).reshape(2,2)
invA = np.linalg.inv(A)
print(invA)
print(np.dot(A, invA))

# 4. Pseudo-inversa
A = np.random.normal(0, 1, (2,2))
l_pinv = np.dot(np.linalg.pinv(A), A)
print(l_pinv)
r_pinv = np.dot(A, np.linalg.pinv(A))
print(r_pinv)






















