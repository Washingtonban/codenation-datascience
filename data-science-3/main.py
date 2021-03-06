#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[3]:


from math import sqrt
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA

from loguru import logger


# In[4]:


# Algumas configurações para o matplotlib.
# %matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[5]:


fifa = pd.read_csv("./data/fifa.csv")


# In[6]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[7]:


# Sua análise começa aqui.


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.
# 

# In[8]:


fifa = fifa.dropna()
def q1():
    pca = PCA(n_components=1)
    pricipal_component = pca.fit_transform(fifa)
    explaned_variance = pca.explained_variance_ratio_
    return float(explaned_variance.round(3)[0])


# In[9]:


q1()


# [Referencia - PCA](https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)

# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[10]:


def q2():
    pca = PCA(n_components=.95)
    pricipal_component = pca.fit_transform(fifa)
    return len(list(pd.DataFrame(pricipal_component).columns))


# In[11]:


q2()


# [Referencia - PCA](https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)

# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[12]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[13]:


def q3():
    pca = PCA(n_components=2)
    pca_fit = pca.fit(fifa)
    return tuple(pca.components_.dot(x).round(3))


# In[14]:


q3()


# [Referencia - PCA](https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)

# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[15]:


def q4():
    selected_variable = 'Age'
    x = fifa.drop(selected_variable, axis=1)
    y = fifa[selected_variable]
    names = fifa.columns

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    model = LinearRegression()
    model.fit(x_train, y_train)

    rfe = RFE(model)
    rfe.fit(x_train, y_train)

    result = pd.DataFrame(sorted(zip(map(lambda x: round(x, 3), model.coef_), names), reverse=True)[:5])
    return list(result[1])


# In[16]:


q4()


# [Referencia - RFE](https://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/)
# 
# 
# 

# In[18]:


def q3():
    pca = PCA(n_components=2)
    pca_fit = pca.fit(fifa)
    return tuple(pca.components_.dot(x).round(3))


# In[19]:


q3()


# [Referencia - PCA](https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)

# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[24]:


def q4():
    selected_variable = 'Overall'
    x = fifa.drop(selected_variable, axis=1)
    y = fifa[selected_variable]

    linear_regression = LinearRegression()
    rfe = RFE(estimator=linear_regression, n_features_to_select=5)
    fit = rfe.fit(x, y)
    features = fit.support_


    return list(x.columns[features])


# In[25]:


q4()


# [Referencia - RFE](https://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/)
# 
# 
# 
