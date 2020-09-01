'''
    Autor: Andressa Gomes Moreira - 402305
    Disciplina: Inteligência Computacional 2020.1
    1° Trabalho: Regressão simples e múltipla
'''

import numpy as np
import matplotlib.pyplot as plt

# Carregando a base de dados
data = open('aerogerador.dat','r')

x = list()     # Variável de entrada x: Velocidade do vento (1° coluna)
y = list()     # Variável de saída y: Potência gerada (2° coluna)

linhas = list()     # Variável auxiliar para receber as linhas da matriz da base de dados
colunas = list()    # Variável auxiliar para receber as colunas da matriz da base de dados

# Dividindo a base de dados em duas variáveis x, y:
for line in data:
    columns = line.split()

    x.append(columns[0]) if (columns[0]) else ''
    y.append(columns[1]) if columns[1] else ''

# Exibir a base de dados "aerogerador.dat" na forma matricial
for aux in range(len(x)):
    linhas = (float(x[aux]), float(y[aux]))
    colunas.append(linhas)
data = np.array(colunas)


# REGRESSÃO LINEAR SIMPLES

'''

    A variável aleatória y pode ser descrita pelo seguinte modelo -> y = B0 + B1 * x
        - y: Variável aleatória                                         
        - B0: intercepto 
        - B1: inclinação

        B1 =  sum(y * x) - (1 / n) * ( sum y) * (sum x) / sum(x^2) - (1/n) * (sum x)^2
        
        B0 = y' - B1 *  x'
            
            -> y' = (1/n) * sum(y)
            -> x' = (1/n) * sum(x)
            
        BO = ((1/n) * sum(y)) - B1 * ((1/n) * sum(x))

'''

# Para determinar os parâmetros desconhecidos B0 e B1 iremos precisar:
n = len(x)  # Quantidade de elementos da amostra => n = len(y)
sum_xy = 0  # Somatório xi*yi
sum_x = 0   # Somatório xi
sum_y = 0   # Somatório yi
sum_x2 = 0  # Somatório (xi**2)

# Realizando os somatórios:
for i in range(n):
    sum_xy += float(x[i])*float(y[i])
    sum_x += float(x[i])
    sum_y += float(y[i])
    sum_x2 += float(x[i])**2

# B1 e B0 são iguais a:
B1 = (sum_xy - ((1/n) * sum_y * sum_x)) / (sum_x2 - ((1/n) * (sum_x**2)))
B0 = ((1/n) * sum_y) - (B1 * ((1/n) * sum_x))

y_reta_aux = list()  # Variável auxiliar para exibir a lista com todos os valores da reta de regressão
linhas1 = list()     # Variável auxiliar para receber as linhas da matriz.
colunas1 = list()    # Variável auxiliar para receber as colunas da matriz.

# Portanto, a reta de regressão será: y_reta = B0 + B1 * x
for i in range(n):
    y_reta = B0 + (B1 * float(x[i]))
    y_reta_aux.append(y_reta)

    linhas1 = (float(x[i]), float(y_reta))
    colunas1.append(linhas1)

new_data = np.array(colunas1)    # Base de dados regressão linear

# =*=*=*=*=**=*=**=**=*=*=*=*==**=**=*=*=*=*=***=**=*=*=*=*=*=*=**=*=**=**=*=*=*=*==**=**=*=*=*=*=***=**=*=*=*=*=*=*=**=*=**=**=*=*=*=*==**=**=*=*=*=*=***=**=*=*
'''
print(f'Base de dados aerogerador.dat: \n {data} \n',)
print(f'B0 = {B0} (Intercepto)\n')
print(f'B1 = {B1} (Inclinação)\n')
print(f'Reta de regressão => {y_reta_aux} \n')
print(f'Coeficiente de determinação => R2 = {R2:.2f} \n')
print(f'Base de dados com os valores de x e com os valores da reta de regressão: \n {new_data} \n')
'''
# =*=*=*=*=**=*=**=**=*=*=*=*==**=**=*=*=*=*=***=**=*=*=*=*=*=*=**=*=**=**=*=*=*=*==**=**=*=*=*=*=***=**=*=*=*=*=*=*=**=*=**=**=*=*=*=*==**=**=*=*=*=*=***=**=*=*

# REGRESSÃO MÚLTIPLA

'''
    Deveremos criar a matriz X correspondente a cada grau do polinômio
        Modelo de regressão linear múltipla com cinco variáveis de entrada:
            -> x1 = x, x2 = x^2, x3 = x^3, x4 = x^4, x5 = x^5
            
        Em forma matricial, o sistema de equações: y = X*β
        As matrizes serão:  
            - Matriz polinômio grau 2: [1, x, x^2]
            - Matriz polinômio grau 3: [1, x, x^2, x^3]
            - Matriz polinômio grau 4: [1, x, x^2, x^3, x^4]
            - Matriz polinômio grau 5: [1, x, x^2, x^3, x^4, x^5]         
'''

# Variáveis que irão receber as linhas e as colunas de cada matriz
linhas_matriz2 = list()
colunas_matriz2 = list()

linhas_matriz3 = list()
colunas_matriz3 = list()

linhas_matriz4 = list()
colunas_matriz4 = list()

linhas_matriz5 = list()
colunas_matriz5 = list()


for i in range(n):
    # Matriz grau 2 => [1, x, x^2]:
    linhas_matriz2 = (1, float(x[i]), float(x[i])**2)
    colunas_matriz2.append(linhas_matriz2)

    # Matriz grau 3 => [1, x, x^2, x^3]:
    linhas_matriz3 = (1, float(x[i]), float(x[i])**2, float(x[i])**3)
    colunas_matriz3.append(linhas_matriz3)

    # Matriz grau 4 => [1, x, x^2, x^3, x^4]:
    linhas_matriz4 = (1, float(x[i]), float(x[i]) ** 2, float(x[i]) ** 3, float(x[i]) ** 4)
    colunas_matriz4.append(linhas_matriz4)

    # Matriz grau 5 => [1, x, x^2, x^3, x^4, x^5]:
    linhas_matriz5 = (1, float(x[i]), float(x[i]) ** 2, float(x[i]) ** 3, float(x[i]) ** 4, float(x[i]) ** 5)
    colunas_matriz5.append(linhas_matriz5)

matriz2 = np.array(colunas_matriz2)  # Matriz polinômio grau 2
matriz3 = np.array(colunas_matriz3)  # Matriz polinômio grau 3
matriz4 = np.array(colunas_matriz4)  # Matriz polinômio grau 4
matriz5 = np.array(colunas_matriz5)  # Matriz polinômio grau 5

'''
    A estimativa de quadrados mínimos de β é dada por:
        - β = (XT * X)^-1 * XT * y
        
    E o modelo de regressão ajustado (preditor) é definido como:
        - y^ = X * β 
'''


# Estimativa de quadrados mínimos de β:
def min_quadrados(grau):
    # Determina qual o grau do polinômio
    if grau == 2:
        X = matriz2
    elif grau == 3:
        X = matriz3
    elif grau == 4:
        X = matriz4
    elif grau == 5:
        X = matriz5

    # Transposta, multiplicação, inversa => (XT * X)^-1:
    XT = X.T                            # TRANSPOSTA
    multip = np.dot(XT, X)              # Multiplicação XT * X
    inversa = np.linalg.inv(multip)     # Inversa (XT * X)^-1

    # Ajustando o y para o formato matricial
    linhas_y = list()
    colunas_y = list()

    for i in range(n):
        linhas_y = [data[i][1]]
        colunas_y.append(linhas_y)

    y_2 = np.array(colunas_y)

    # Logo,  estimativa de quadrados mínimos de β é igual a:
    multip1 = np.dot(inversa, XT)
    beta = np.dot(multip1, y_2)

    return beta


# Resultado da regressão polinomial de grau 2
b_2 = min_quadrados(2)                  # Estimativa de quadrados mínimos de β (Grau 2)
y_2 = np.dot(matriz2, b_2)              # Modelo de regressão ajustado (Grau 2)

# Resultado da regressão polinomial de grau 3
b_3 = min_quadrados(3)                  # Estimativa de quadrados mínimos de β (Grau 3)
y_3 = np.dot(matriz3, b_3)              # Modelo de regressão ajustado (Grau 3)

# Resultado da regressão polinomial de grau 4
b_4 = min_quadrados(4)                  # Estimativa de quadrados mínimos de β (Grau 4)
y_4 = np.dot(matriz4, b_4)              # Modelo de regressão ajustado (Grau 4)

# Resultado da regressão polinomial de grau 5
b_5 = min_quadrados(5)                  # Estimativa de quadrados mínimos de β (Grau 5)
y_5 = np.dot(matriz5, b_5)              # Modelo de regressão ajustado (Grau 5)

print('Valores para os coeficientes B:')
print(f'Estimativa de quadrados mínimos de β (Grau 2) = {b_2}')
print(f'Estimativa de quadrados mínimos de β (Grau 3) = {b_3}')
print(f'Estimativa de quadrados mínimos de β (Grau 4) = {b_4}')
print(f'Estimativa de quadrados mínimos de β (Grau 5) = {b_5}')


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
SQe = 0                 # Soma de quadrados dos resíduos
Syy = 0
media = sum_y / n       # Variável auxiliar para o cálculo de Syy


# Função para determinar os coeficientes de determinação na Regressão Linear e na Múltipla
def coef_determinacao(y_aux, grau, SQe=0, Syy=0):
    p = grau + 1
    for i in range(n):
        SQe += ((float(y[i])) - (float(y_aux[i]))) ** 2
        Syy += ((float(y[i])) - media) ** 2

    R2 = 1 - (SQe / Syy)                                # Resultado do coeficiente de determinação.
    R2_aj = 1 - ((SQe / (n - p)) / (Syy / (n - 1)))     # Resultado do coeficiente de determinação ajustado.

    return R2, R2_aj


#Resultado do coeficiente de determinação (Regressão Linear)

print('\nCoeficientes de Determinação: ')

R2_linear, _ = coef_determinacao(y_reta_aux, 1)
print(f'\nRegressão Linear: ')
print(f'Coeficiente de determinação (Regressão Linear) => R2 = {R2_linear}\n')

# Resultado dos coeficientes de determinação na Regressão Múltipla
#Grau 2
R2_polin2, R2_polin2_aj = coef_determinacao(y_2, 2)
print(f'Regressão Polinomial: ')
print(f'Coeficiente de determinação (Grau 2) => R2 = {R2_polin2}')
print(f'Coeficiente de determinação ajustado (Grau 2) => R2_aj = {R2_polin2_aj}\n')

#Grau 3
R2_polin3, R2_polin3_aj = coef_determinacao(y_3, 3)
print(f'Coeficiente de determinação (Grau 3) => R2 = {R2_polin3}')
print(f'Coeficiente de determinação ajustado (Grau 3) => R2_aj = {R2_polin3_aj}\n')

#Grau 4
R2_polin4, R2_polin4_aj = coef_determinacao(y_4, 4)
print(f'Coeficiente de determinação (Grau 4) => R2 = {R2_polin4}')
print(f'Coeficiente de determinação ajustado (Grau 4) => R2_aj = {R2_polin4_aj}\n')

#Grau 5
R2_polin5, R2_polin5_aj = coef_determinacao(y_5, 5)
print(f'Coeficiente de determinação (Grau 5) => R2 = {R2_polin5}')
print(f'Coeficiente de determinação ajustado (Grau 5) => R2_aj = {R2_polin5_aj}\n')


# GRÁFICOS

x_aux = list()
y_aux = list()
for i in range(n):
    x_aux.append(float(x[i]))
    y_aux.append(float(y[i]))


#Gráfico 1 - Regressão Linear Simples
plt.figure(1)
plt.title(f'Regressão Linear - Aerogerador - R2 = {R2_linear}')
plt.xlabel('Velocidade do vento x')
plt.ylabel('Potência gerada y')

plt.plot(x_aux, y_reta_aux, color='red')
plt.scatter(x_aux, y_aux, color='blue')
plt.grid(True)
plt.show()

#Gráficos - Regressão Polinomial
def grafico_poli(grau, matriz, y_i, R2_polin, R2_polin_aj):
    plt.figure(grau)
    plt.scatter(x_aux, y_aux, color='blue')
    plt.plot(matriz[0:, 1:2], y_i[0:, 0:1], color='red')

    plt.ylabel('Potência')
    plt.xlabel('Velocidade do Vento')
    plt.title(f'Regressão Polinomial (Grau {grau}) - R2 = {R2_polin} - R2_aj= {R2_polin_aj}')
    plt.grid(True)
    plt.show()


grafico_poli(2, matriz2, y_2, R2_polin2, R2_polin2_aj)
grafico_poli(3, matriz3, y_3, R2_polin3, R2_polin3_aj)
grafico_poli(4, matriz4, y_4, R2_polin4, R2_polin4_aj)
grafico_poli(5, matriz5, y_5, R2_polin5, R2_polin5_aj)