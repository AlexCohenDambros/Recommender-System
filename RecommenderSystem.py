import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import trunc


dataset = pd.read_csv("ratings_Electronics.csv", names=[
                      "IdUser", "IdProduct", "Rating", "TimesTamp"], sep=",")


# Exibe os dados do cabeçalho
print("Display head data: \n\n{}".format(dataset.head()))

# Substituindo todos os valores de NaN por 0
dataset = dataset.fillna(0)

# Verifique se há valores NaN
print(f'\nNumber of missing values across columns:\n{dataset.isnull().sum()}')

# Shape dos dados
print("\nShape of the data: {}".format(dataset.shape))

# Verificarndo os tipos dos dados
print("\nData Types: \n{}".format(dataset.dtypes))

# Resumo
summary = dataset.describe()['Rating'].T
print("\nSummary: \n\n{}".format(summary))

# Classificações mínimas e máximas
print('\nMinimum and maximum ratings: \nMinimum rating is: {:d} \nMaximum rating is: {:d}'.format(
    trunc(dataset.Rating.min()), trunc(dataset.Rating.max())))


# Verifique a distribuição da classificação
freq = pd.Series(dataset.Rating.value_counts())
freq.plot.bar(figsize=(12, 6))
plt.ticklabel_format(style='plain', axis='y')
plt.suptitle('Distribuição da classificação', fontsize=25)
plt.xlabel("Avaliação")
plt.ylabel("Quantidade")

# Usuários e produtos únicos
print("="*50)
print("\n ------ Unique Users and products ------")
print("\nTotal no of ratings: ", dataset.shape[0])
print("Total No of Users: ", len(np.unique(dataset.IdUser)))
print("Total No of products: ", len(np.unique(dataset.IdProduct)))


# Análise da classificação dada pelo usuário
print("\nAnalysis of rating given by the user:\n")
no_of_rated_products_per_user = dataset.groupby(
    by='IdUser')['Rating'].count().sort_values(ascending=False)

print(f"\n{no_of_rated_products_per_user}")

print(f"\n{no_of_rated_products_per_user.describe()}")

# =========== Recomendação baseada em popularidade ===========

# O sistema de recomendação baseado em popularidade funciona com a tendência. Ele basicamente usa os itens que estão em tendência no momento.
# Por exemplo, se qualquer produto que geralmente é comprado por cada novo usuário, há chances de que ele sugira esse item para o usuário que acabou de se inscrever.

# O problema com o sistema de recomendação baseado em popularidade é que a personalização não
# está disponível com este método, ou seja, mesmo que você conheça o comportamento do usuário, não pode recomendar itens de acordo.

# Obtendo um novo dataframe que contém usuários que deram 15 ou mais avalaições
new_Dataset_Rating = dataset.groupby("IdProduct").filter(
    lambda x: x['Rating'].count() >= 15)

# Média de avaliações por produto
print("\nAverage of rating per product: \n")
print(new_Dataset_Rating.groupby('IdProduct')['Rating'].mean())

# Quantidade total de avaliações por produto
print("\nTotal no of rating for product: \n")
print(new_Dataset_Rating.groupby('IdProduct')['Rating'].count().sort_values(ascending=False))

plt.show()
