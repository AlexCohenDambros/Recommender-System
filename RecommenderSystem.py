import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import trunc
import MachineLearning

dataset = pd.read_csv("ratings_Electronics.csv", names=[
                      "IdUser", "IdProduct", "Rating", "TimesTamp"], sep=",")


# Exibe os dados do cabeçalho
print("Exibir dados(head): \n\n{}".format(dataset.head()))

# Substituindo todos os valores de NaN por 0
dataset = dataset.fillna(0)

# Verifique se há valores NaN
print(f'\nValores ausentes nas colunas:\n{dataset.isnull().sum()}')

# Shape
print("\nShape: {}".format(dataset.shape))

# Verificando os tipos dos dados
print("\nTipos dos dados: \n{}".format(dataset.dtypes))

# sumário
summary = dataset.describe()['Rating'].T
print("\nSumário: \n\n{}".format(summary))

# Classificações mínimas e máximas
print('\nClassificações mínimas e máximas: \nRating mínimo é: {:d} \nRating máximo é: {:d}'.format(
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
print("\n ------ Usuários e produtos únicos ------")
print("\nNº total de avaliações: ", dataset.shape[0])
print("Nº total de usuários: ", len(np.unique(dataset.IdUser)))
print("Nº total de produtos: ", len(np.unique(dataset.IdProduct)))


# Análise da classificação dada pelo usuário
print("\nAnálise da classificação dada pelo usuário:")
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
print("\nMédia de avaliações por produto: \n")
print(new_Dataset_Rating.groupby('IdProduct')['Rating'].mean())

# Quantidade total de avaliações por produto
print("\nQuantidade total de avaliações por produto: \n")
print(new_Dataset_Rating.groupby('IdProduct')[
      'Rating'].count().sort_values(ascending=False))

ratings_mean_count = pd.DataFrame(
    new_Dataset_Rating.groupby('IdProduct')['Rating'].mean())
ratings_mean_count['rating_counts'] = pd.DataFrame(
    new_Dataset_Rating.groupby('IdProduct')['Rating'].count())
print(ratings_mean_count.head())

# produtos mais populares
popular_products = pd.DataFrame(
    new_Dataset_Rating.groupby('IdProduct')['Rating'].count())
mostPopular = popular_products.sort_values('Rating', ascending=False)
mostPopular.head(10).plot(kind='bar', figsize=(12, 6))

MachineLearning.machineLearning(new_Dataset_Rating)

plt.show()
