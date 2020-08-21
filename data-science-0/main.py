#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[310]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[42]:





# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[7]:


def q1():
    result = black_friday.shape
    return result


# In[271]:


q1()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[282]:


def q2():
    return int(black_friday[(black_friday['Gender'] == 'F') & (black_friday['Age'] == '26-35')]['User_ID'].count())


# In[283]:


q2()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[48]:


def q3():
    result = black_friday.User_ID.nunique()
    return result


# In[284]:


q3()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[333]:


def q4():
    result = black_friday.dtypes.nunique()
    return result


# In[334]:


q4()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[293]:


df_all =  black_friday.shape[0]
df_notnull = black_friday.dropna().shape[0]
result = (df_all - df_notnull) / df_all


# In[303]:


def q5():
    df_all =  black_friday.shape[0]
    df_notnull = black_friday.dropna().shape[0]
    result = (df_all - df_notnull) / df_all
    return result


# In[304]:


q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[335]:


def q6():
    result = int(black_friday.isnull().sum().sort_values()[-1])
    return result


# In[336]:


q6()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[183]:


def q7():
    result = black_friday.Product_Category_3.value_counts().index[0]
    return result


# In[307]:


q7()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[337]:


def q8():
    normalized_df=(black_friday.Purchase-black_friday.Purchase.min())/(black_friday.Purchase.max()-black_friday.Purchase.min())
    result = float(normalized_df.mean())
    return result


# In[338]:


q8()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[220]:


def q9():
    normalized_df=(black_friday.Purchase - black_friday.Purchase.mean())/black_friday.Purchase.std()
    result = int(((normalized_df >= -1) & (normalized_df <= 1)).sum())
    return result


# In[221]:


q9()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).
# 

# In[331]:


def q10():
    # Retorne aqui o resultado da questão 10.
    category_2 = black_friday.Product_Category_2 - black_friday.Product_Category_2.dropna()
    category_3 = black_friday.Product_Category_3 - black_friday.Product_Category_3.dropna()
    result = bool(np.isin(category_2, category_3).all())
    return result


# In[332]:


q10()



# In[ ]:




