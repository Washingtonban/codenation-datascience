#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm


# In[9]:


# %matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[10]:


athletes = pd.read_csv("./data/athletes.csv")


# In[11]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[12]:


# Sua análise começa aqui.


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[13]:


def q1():
    significance = 0.05
    height_serie = get_sample(athletes, 'height', 3000)
    return bool(sct.shapiro(height_serie)[1] > significance)


# In[14]:


q1()


# [Referência - shapiro](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)

# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?

# In[15]:


height_serie = get_sample(athletes, 'height', 3000)
height_serie.plot(kind='hist', y='Frequência', x='Distribuição', bins=25)
plt.title('Distribuição das alturas normalizadas')
plt.show()


# [Referencia - Matplotlib](http://queirozf.com/entries/pandas-dataframe-plot-examples-with-matplotlib-pyplot)

# * Plote o qq-plot para essa variável e a analise.

# In[16]:


sm.qqplot(height_serie, fit=True ,line='45')
plt.show()


# [Referencia - qq-plot](https://www.statsmodels.org/stable/generated/statsmodels.graphics.gofplots.qqplot.html)

# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[17]:


def q2():
    significance = 0.05
    height_serie = get_sample(athletes, 'height', 3000)
    return bool(sct.jarque_bera(height_serie)[1] > significance)


# In[18]:


q2()


# [Referência - Jarque Bera](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.jarque_bera.html)

# __Para refletir__:
# 
# * Esse resultado faz sentido?

# In[18]:





# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[19]:


def q3():
    significance = 0.05
    height_serie = get_sample(athletes, 'weight', 3000)
    return bool(sct.normaltest(height_serie)[1] > significance)


# In[20]:


q3()


# [Referência - SciPy.org](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.normaltest.html)

# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?

# In[21]:


height_serie = get_sample(athletes, 'weight', 3000)
height_serie.plot(kind='hist', y='Frequência', x='Distribuição', bins=25)
plt.title('Distribuição dos pesos normalizadas')
plt.show()


# * Um _box plot_ também poderia ajudar a entender a resposta.

# In[22]:


height_serie = get_sample(athletes, 'weight', 3000)
height_serie.plot(kind='box', vert=False)
plt.title('Boxplot dos pesos')
plt.show()


# [Referência - Boxplot](https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.boxplot.html)

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[23]:


def q4():
    significance = 0.05
    height_serie = np.log(get_sample(athletes, 'height', 3000))
    return bool(sct.shapiro(height_serie)[1] > significance)


# In[24]:


q4()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[33]:


def q5():
    bra = athletes[athletes['nationality'] == 'BRA'].height.dropna()
    usa = athletes[athletes['nationality'] == 'USA'].height.dropna()
    significance = 0.05
    statistic, pvalue = sct.stats.ttest_ind(bra, usa)
    return bool(pvalue > significance)


# In[34]:


q5()


# [Referência - ttest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)

# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[35]:


def q6():
    bra = athletes[athletes['nationality'] == 'BRA'].height.dropna()
    can = athletes[athletes['nationality'] == 'CAN'].height.dropna()
    significance = 0.05
    statistic, pvalue = sct.stats.ttest_ind(bra, can)
    return bool(pvalue > significance)


# In[36]:


q6()


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[44]:


def q7():
    usa = athletes[athletes['nationality'] == 'USA'].height.dropna()
    can = athletes[athletes['nationality'] == 'CAN'].height.dropna()
    statistic, pvalue = sct.stats.ttest_ind(usa, can, equal_var=False)
    return float(pvalue.round(8))


# In[45]:


q7()


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?
