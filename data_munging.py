import pandas as pd
import numpy as np

df = pd.read_csv('train.csv', index_col = 'Loan_ID')

#definindo uma função apply para verificar a existência de valores nulos
df.apply(lambda x: sum(x.isnull()), axis=0)

#preenchendo os valores nulos de LoanAmount
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)

#em relação a coluna Self_Employed, realizando uma análise de frequência,
#percebe-se que 86% dos dados não são autônomos, fazendo com que seja 
#seguro substituir os valores nulos por novos dados NÃO autônomos
df['Self_Employed'].value_counts()

df['Self_Employed'].fillna('No', inplace=True)

#definindo um pivoteamento na tabela para visualizar LoanAmount e Self_Employed
#baseado na coluna Education
table = df.pivot_table(values='LoanAmount', index='Self_Employed', columns='Education', aggfunc=np.median)

#definindo uma função para retornar os valores desse pivoteamento
def fage(x):
    return table.loc[x['Self_Employed'], x['Education']]

#substituir os valores nulos em LoanAmount
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)

#tratando os valores extremos em LoanAmount e ApplicationIncome
#realizando Log Transformation em LoanAmount para normalizar os valores discrepantes
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20) #valores normalizados
df['LoanAmount'].hist(bins=20)  #valores originais

#tratando ApplicantIncome
#Uma boa intuição é relacionar ApplicantIncome com CoapplicantIncome, isso porque
#mesmo um candidato tendo uma renda baixa, pode ser que apoie fortemente os 
#co-candidatos, logo é uma boa ideia combina-los e realizar uma transofmração logarítmica
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20) #normalizado
df['TotalIncome'].hist(bins=20) #original

#preenchendo o restante das variáveis nulas no dataset
df['Gender'].fillna(df['Gender'].mode()[0], inplace = True)
df['Married'].fillna(df['Married'].mode()[0], inplace = True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace = True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace = True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace = True)