import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

#transformando as variáveis categóricas em numéricas
var_mod = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
label_encoder = LabelEncoder()
for i in var_mod:
    df[i] = label_encoder.fit_transform(df[i])
    
df.dtypes

#criando uma função para o modelo de classificação e verificação de performance
def classification_model(model, data, predictors, outcome):
    #criando o modelo
    model.fit(data[predictors], data[outcome])
    
    #realizando as previsões
    predictions = model.predict(data[predictors])
    
    #mostrando a acurácia
    acc = metrics.accuracy_score(predictions, data[outcome])
    print('Accuracy: %s' % '{0: .3%}'.format(acc))
    '''
    #realizando k-fold cross-validation com 5 folds
    kf = KFold(data.shape[0])
    error = []
    for train, test in kf:
        #encontrando os atributos
        train_predictors = (data[predictors].iloc[train, :])
    '''    
        #encontrando as classes
        #train_target = data[outcome].iloc[train]
        
    #treinando com atributos e classes
    #model.fit(train_predictors, train_target)
    
    #gravando os erros de validação cruzada
    #error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    
    #print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
    
    #treinando novamente o modelo
    #model.fit(data[predictors], data[outcome])
    
#criando o primeiro modelo de regressão logística utilizando a variável Loan_Status como análise
outcome_var = 'Loan_Status'
model = LogisticRegression()
prediction_var = ['Credit_History']
classification_model(model, df, prediction_var, outcome_var)
#obtemos 80.945% de acc

#realizando o mesmo processo utilizando uma combinação de variáveis
prediction_var = ['Credit_History', 'Education', 'Married', 'Self_Employed', 'Property_Area']
classification_model(model, df, prediction_var, outcome_var)

#modelo de árvore de decisão
model = DecisionTreeClassifier()
prediction_var = ['Credit_History', 'Gender', 'Married', 'Education']
classification_model(model, df, prediction_var, outcome_var)

#a variável Credit_History foi dominante, logo, vamos testar com outras variáveis numéricas
prediction_var = ['Credit_History', 'Loan_Amount_Term', 'LoanAmount_log']
classification_model(model, df, prediction_var, outcome_var)
#com isso obtivemos uma acc de 88.925%, porém, o modelo está com overfitting

#modelo com Random Forest
model = RandomForestClassifier(n_estimators=100)
prediction_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','TotalIncome_log']
classification_model(model, df, prediction_var, outcome_var)
#a acc foi de 100%, um caso claro de overfitting. Logo, podemos reduzir o número de atributos ou melhorar os parâmetros do modelo

#verificando a matriz de importância dos atributos
featimp = pd.Series(model.feature_importances_, index=prediction_var).sort_values(ascending=False)
print(featimp)

#utilizando os 5 primeiros atributos e melhorando os parâmetros do modelo
model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
prediction_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
classification_model(model, df, prediction_var, outcome_var)
#obtemos acc de 83.388%















