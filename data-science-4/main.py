#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk


from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_selector as selector
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline


# In[2]:


# Algumas configurações para o matplotlib.
# %matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


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

# In[5]:


# Fazendo os ajustes necessários no dataset

countries['Country'] = countries['Country'].str.strip() # Retirando espaços em branco da coluna Country
countries['Region'] = countries['Region'].str.strip() # Retirando espaços em branco da coluna Region

countries = countries.apply(lambda x: x.replace(',', '.', regex = True)) # Mudando separador de virgula para ponto

countries.iloc[:,2:] = countries.iloc[:,2:].apply(pd.to_numeric) #Convertendo strings em números, retirando as duas primeiras colunas de texto


# ## Inicia sua análise a partir daqui

# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[6]:


def q1():
    return sorted(countries['Region'].unique())


# In[7]:


q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[8]:


def q2():
    dis = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    countries['Pop_density'] = dis.fit_transform(countries[['Pop_density']])
    return int(countries['Pop_density'][countries['Pop_density'] == 9.0].count())


# In[9]:


q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[10]:


def q3():
    onehot = OneHotEncoder(sparse = False).fit_transform(countries[['Region', 'Climate']].fillna(0))
    return onehot.shape[1]


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


def q4():
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    pipeline = make_pipeline(imputer, scaler)
    
    countries_num = countries.select_dtypes(include=[np.number])
    pipeline.fit(countries_num)
    transformed_test_country = pipeline.transform([test_country[2:]])
    arable = transformed_test_country[:, countries_num.columns.get_loc("Arable")]
    
    return float(np.around(arable.item(),3))


# In[14]:


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

# In[28]:


def q5():
    q1 = countries['Net_migration'].quantile(0.25)
    q3 = countries['Net_migration'].quantile(0.75)
    iqr = q3 - q1  
    
    outliers_abaixo = countries['Net_migration'][countries['Net_migration'] < (q1 - 1.5 * iqr)].count()
    outliers_acima = countries['Net_migration'][countries['Net_migration'] > (q3 + 1.5 * iqr)].count()    
    removeria = False 
   
    return (int(outliers_abaixo), int(outliers_acima), removeria)


# In[29]:


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
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[17]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[18]:


def q6():
    count_vec = CountVectorizer()
    newsgroup_counts = count_vec.fit_transform(newsgroup.data)
    word_index = count_vec.vocabulary_.get(u'phone')
    
    count_list = newsgroup_counts.sum(axis=0)
    phone_count = count_list[0,word_index] 
    
    return int(phone_count)


# In[19]:


q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[24]:


def q7():
    vec = TfidfVectorizer (use_idf=True)
    newsgroup_counts = vec.fit_transform(newsgroup.data)
    word_index = vec.vocabulary_.get(u'phone')
    
    tfidf_list = newsgroup_counts.sum(axis=0)
    
    word_tdidf = tfidf_list[0,word_index] 
    
    return float(word_tdidf.round(3))


# In[25]:


q7()

