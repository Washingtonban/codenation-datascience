#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[72]:


from IPython.core.pylabtools import figsize
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk


# In[2]:


# Algumas configurações para o matplotlib.
# %matplotlib inline


figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv("./data/countries.csv", thousands='.', decimal=',')


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[6]:


def q1():
    return list(sorted(countries.Region.str.strip().unique()))


# In[7]:


q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[8]:


def q2():
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal',  strategy='quantile')
    return int((discretizer.fit_transform(countries[['Pop_density']]) >= 9).sum())


# In[9]:


q2()


# 
# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[10]:


def q3():
    return int(pd.get_dummies(countries[['Region', 'Climate']].fillna('NaN')).shape[1])


# In[11]:


q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[12]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[13]:


def split_df_type(df):
    columns = df.columns
    i64 = []
    f64 = []
    objs = []
    for column in columns:
        if countries[column].dtypes == 'int64':
            i64.append(column)
        elif countries[column].dtypes == 'float64':
            f64.append(column)
        else:
            objs.append(column)
    return i64, f64, objs


# In[14]:


i64, f64, objs = split_df_type(countries)


# In[15]:


numerics_columns = i64 + f64


# In[90]:


def q4():
    pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                           ('scaler', StandardScaler())])
    pipe.fit(countries[numerics_columns])
    pipe_transform = pipe.transform(np.asarray(test_country[2:]).reshape(1,-1))
    df = pd.DataFrame(pipe_transform, columns=numerics_columns)
    return round(df.Arable[0].item(), 3)


# In[91]:


q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[43]:


def q5():
    q1, q3 = countries.Net_migration.dropna().quantile([0.25, 0.75])
    iqr = q3 - q1
    lup, ldown = q3 + 1.5 * iqr, q1 - 1.5 * iqr
    outliers_abaixo = countries.Net_migration[(countries.Net_migration > lup)]
    outliers_acima = countries.Net_migration[countries.Net_migration < ldown]
    removeria = False
    return (outliers_abaixo.shape[0], outliers_acima.shape[0], removeria)


# In[44]:


q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```

# In[78]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[68]:


def q6():
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(newsgroup.data)
    return int(x[:, vectorizer.vocabulary_['phone']].sum())


# In[69]:


q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[88]:


def q7():
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(newsgroup.data)
    return round(float(x[:, vectorizer.vocabulary_['phone']].sum()), 3)


# In[89]:


q7()


# In[ ]:




