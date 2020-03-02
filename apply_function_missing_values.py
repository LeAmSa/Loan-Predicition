import pandas as pd
import numpy as np
from scipy.stats import mode


data = pd.read_csv('train.csv', index_col = 'Loan_ID')

#visualizando todas as linhas das colunas Gender, Education e Loan_Status que contém mulheres não graduadas que conseguiram empréstimo
data.loc[(data['Gender'] == 'Female') & (data['Education'] == 'Not Graduate') 
            & (data['Loan_Status'] == 'Y'), ['Gender', 'Education', 'Loan_Status']]

#criando uma APPLY funtion para encontrar valores faltantes

#criando a função que encontra esses valores
def num_missing(x):
    return sum(x.isnull())

#verificando por coluna
print('Valores nulos por coluna:')
print(data.apply(num_missing, axis=0)) #axis = 0 faz com que seja aplicada em cada coluna

#verificando nas linhas
print('Valores nulos por linhas:')
print(data.apply(num_missing, axis=1))

#determinando qual será o ojbeto a ser modificado
mode(data['Gender'])
mode(data['Gender']).mode[0]

#acrescentando os valores
data['Gender'].fillna(mode(data['Gender']).mode[0], inplace=True)
data['Married'].fillna(mode(data['Married']).mode[0], inplace=True)
data['Self_Employed'].fillna(mode(data['Self_Employed']).mode[0], inplace=True)