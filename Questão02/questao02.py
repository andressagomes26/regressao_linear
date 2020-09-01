'''
    Autor: Andressa Gomes Moreira - 402305
    Disciplina: Inteligência Computacional 2020.1
    1° Trabalho: Regressão simples e múltipla
    Questão 02
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
    A base de dados deve ser organizada da seguinte forma: 
        - Variáveis regressoras (x1 e x2): 1° e 2° colunas.
        - Variável dependete (y): 3° coluna.
        
        
        D =[122 139 0.115,       
            114 126 0.120,       
            086 090 0.105,       
            134 144 0.090,       
            146 163 0.100,       
            107 136 0.120,       
            068 061 0.105,       
            117 062 0.080,       
            071 041 0.100,        
            098 120 0.115]
'''

# Organizando a base de dados:
x = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [122, 114, 86, 134, 146, 107, 68, 117, 71, 98],
    [139, 126, 90, 144, 163, 136, 61, 62, 41, 120]
]

y = [0.115, 0.120, 0.105, 0.090, 0.100, 0.120, 0.105, 0.080, 0.100, 0.115]

#Variáveis auxiliares que irão receber as linhas e as colunas da matriz
n = len(y)                      #  Quantidade de valores na amostra
linhas_X = list()
colunas_X = list()

linhas_y = list()
colunas_y = list()

# Ajustando a base de dados na forma matricial
for i in range(n):
    linhas_X = (1, float(x[1][i]), float(x[2][i]))
    colunas_X.append(linhas_X)

    linhas_y = [y[i]]
    colunas_y.append(linhas_y)

X = np.array(colunas_X)         # Matriz com valores das variáveis regressoras (x1 e x2)
Y = np.array(colunas_y)         # Matriz com valores da ariável dependete (y)

# ESTIMATIVA DE QUADRADOS MÍNIMOS

'''
    A estimativa de quadrados mínimos de β é dada por:
        - β = (XT * X)^-1 * XT * y

    E o modelo de regressão ajustado (preditor) é definido como:
        - y^ = X * β
'''

# Função que irá estipular a estimativa de quadrados mínimos β:
def min_quadrados():
    # Transposta, multiplicação, inversa => (XT * X)^-1:
    XT = X.T                            # TRANSPOSTA
    multip = np.dot(XT, X)              # Multiplicação (XT * X)
    inversa = np.linalg.inv(multip)     # Inversa (XT * X)^-1

    multip1 = np.dot(inversa, XT)       # Multiplicação ((XT * X)^-1 * XT
    beta = np.dot(multip1, Y)           # Resultado: β = (XT * X)^-1 * XT * y

    return beta                         # Retorna a estimativa de quadrados mínimos β


# Resultado da Regressão
beta = min_quadrados()                # Estimativa de quadrados mínimos de β
y_beta = np.dot(X, beta)              # Modelo de regressão ajustado

# COEFICIENTES DE DETERMINAÇÃO

'''
    O coeficiente de determinação R2 é usado para determinar a
    adequação de um modelo de regressão. Podemos defini-lo como:

        -> R2 = 1 - (SQe / Syy)

            -> SQe = sum(yi - y^)^2 
            -> Syy = sum(yi - y')^2

    O Coeficiente de determinação ajustado R2aj:

        -> R2aj = 1 - ((SQe / n-p) / (Syy / n-1))

                -> SQe = sum(yi - y^)^2 
                -> Syy = sum(yi - y')^2
                -> p = k + 1
'''

# Variáveis que serão utilizadas no cálculo do coeficiente de determinação
SQe = 0             # Soma de quadrados dos resíduos
Syy = 0
media = sum(Y/n)    # Variável auxiliar para o cálculo de Syy


# Função para determinar os coeficientes de determinação
def coef_determinacao(y_aux, k, SQe=0, Syy=0):
    p = k + 1
    for i in range(n):
        SQe += ((float(Y[i])) - (float(y_aux[i]))) ** 2
        Syy += ((float(Y[i])) - media) ** 2

    R2 = 1 - (SQe / Syy)                                # Resultado do coeficiente de determinação.
    R2_aj = 1 - ((SQe / (n - p)) / (Syy / (n - 1)))     # Resultado do coeficiente de determinação ajustado.

    return R2, R2_aj

R2, R2_aj = coef_determinacao(y_beta, 2)

print(f'R2 = {R2}')
print(f'R2_aj = {R2_aj}\n')

print(f'β = {beta}\n')
print(f'β0 = {beta[0]}')
print(f'β1 = {beta[1]}')
print(f'β2 = {beta[2]}')


# GRÁFICO

#Ajustando para os valores em uma lista
x1 = list()
x2 = list()

for i in range(n):
    x1.append(float(X[i][1]))
    x2.append(float(X[i][2]))

fig = plt.figure()
ax = Axes3D(fig)

resx = x1
resy = x2
resz = y_beta

ax.scatter(resx, resy, resz, color='red')

plt.title(f'Regressão Mútipla R2 = {R2} e R2_aj = {R2_aj}')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')

plt.show()

