import pandas as pd
import numpy as np
from requests import delete

# Treinamento de classificadores
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

import warnings
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

np.random.seed(42)

pd.options.mode.chained_assignment = None

# ======== Parametros das funcoes de Machine Learning ========

parametrosKNN = [
    {
        'n_neighbors': range(1, 10, 1),
        'weights': ['uniform', 'distance'],
        'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2],
        'leaf_size': range(2, 20, 2)
    }
]

parametersSVM = [
    {
        'C': [1, 5, 10, 50, 100, 150, 500],
        'gamma': [0.1, 0.01, 0.001, 0.0001, 'scale'],
        'kernel':['rbf', 'poly', 'linear', 'sigmoid', 'precomputed'],
        'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    }
]

parametersRandomForest = [
    {
        'n_estimators': range(80, 200, 20),
        'max_depth': range(3, 30, 3),
        'min_samples_split': range(5, 25, 5),
        'criterion': ['gini', 'entropy']
    }
]

parametersBagging = [
    {
        'n_estimators': range(80, 200, 40),
        'base_estimator': [None, DecisionTreeClassifier(criterion='entropy', max_depth=5), KNeighborsClassifier(n_neighbors=3), KNeighborsClassifier(n_neighbors=3, weights='distance')]
    }
]

parametersMLP = [
    {
        'solver': ['adam', 'sgd'],
        'hidden_layer_sizes': [4, 7, (6, 8)],
        'max_iter': [100, 300, 500, 800, 1300],
    }
]


def plotResultados(matrix):
    print("Confusion Matrix:")
    print(matrix)


def k_Nearest_Neighbors(parameters, folds, X_train, X_test, y_train, y_test):

    model = KNeighborsClassifier()

    model = GridSearchCV(model, parameters,
                         scoring='accuracy', cv=folds,  n_jobs=-1)

    model = model.fit(X_train, y_train)

    best_parameters = model.best_params_

    # test using test base
    predicted = model.predict(X_test)

    result = model_selection.cross_val_score(
        model, X_train, y_train, cv=folds, n_jobs=-1)

    # calculates accuracy based on test
    score = result.mean()

    # calculate the confusion matrix
    matrix = confusion_matrix(y_test, predicted)

    print("\nFinished!")

    return score, matrix, best_parameters


def supportVectorMachine(parameters, folds, X_train, X_test, y_train, y_test):

    model = SVC(probability=True, random_state=42)

    model = GridSearchCV(model, parameters,
                         scoring='accuracy', cv=folds,  n_jobs=-1)

    model = model.fit(X_train, y_train)

    best_parameters = model.best_params_

    # test using test base
    predicted = model.predict(X_test)

    result = model_selection.cross_val_score(
        model, X_train, y_train, cv=folds, n_jobs=-1)

    # calculates accuracy based on test
    score = result.mean()

    # calculate the confusion matrix
    matrix = confusion_matrix(y_test, predicted)

    print("\nFinished!")

    return score, matrix, best_parameters


def baggingClassifier(parameters, folds, X_train, X_test, y_train, y_test):

    clfa = BaggingClassifier(random_state=42)

    clfa = GridSearchCV(
        clfa, parameters, scoring='accuracy', cv=folds, n_jobs=-1)

    clfa = clfa.fit(X_train, y_train)

    best_parameters = clfa.best_params_

    # testa usando a base de testes
    predicted = clfa.predict(X_test)

    result = model_selection.cross_val_score(clfa, X_train, y_train, cv=folds)

    # calcula a acuracia na base de teste
    score = result.mean()

    # calcula a matriz de confusao
    matrix = confusion_matrix(y_test, predicted)

    print("\nFinished!")

    return score, matrix, best_parameters


def randomForestClassifier(parameters, folds, X_train, X_test, y_train, y_test):

    clfa = RandomForestClassifier(random_state=42)

    clfa = GridSearchCV(
        clfa, parameters, scoring='accuracy', cv=folds, n_jobs=-1)

    clfa = clfa.fit(X_train, y_train)

    best_parameters = clfa.best_params_

    # testa usando a base de testes
    predicted = clfa.predict(X_test)

    result = model_selection.cross_val_score(clfa, X_train, y_train, cv=folds)

    # calcula a acuracia na base de teste
    score = result.mean()

    # calcula a matriz de confusao
    matrix = confusion_matrix(y_test, predicted)

    print("\nFinished!")

    return score, matrix, best_parameters


def mlpClassifier(parameters, folds, X_train, X_test, y_train, y_test):

    # Treina o classificador
    clfa = MLPClassifier(random_state=42)

    clfa = GridSearchCV(
        clfa, parameters, scoring='accuracy', cv=folds, n_jobs=-1)

    clfa = clfa.fit(X_train, y_train)

    best_parameters = clfa.best_params_

    # testa usando a base de testes
    predicted = clfa.predict(X_test)

    result = model_selection.cross_val_score(clfa, X_train, y_train, cv=folds)

    # calcula a acuracia na base de teste
    score = result.mean()

    # calcula a matriz de confusao
    matrix = confusion_matrix(y_test, predicted)

    print("\nFinished!")

    return score, matrix, best_parameters


def machineLearning(dataset_main):
    melhoresResultados = {}
    print("="*50)
    
    dataset_main = dataset_main.head(20000)
    
    dataset = dataset_main.copy()
    
    dataset["IdUser"] = dataset["IdUser"].factorize()[0]
    dataset["IdProduct"] = dataset["IdProduct"].factorize()[0] 

    valores = dataset.values
    y = valores[:, 2]
    dataset = dataset.drop(['Rating'], axis=1)
    novos_valores = dataset.values
    X = novos_valores[:, 0:3]

    # separa teste e treino
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42, stratify=y)

    folds = 10

    print("\n==== Executando --> KNN ====")
    score, matrix, best_parameters = k_Nearest_Neighbors(
        parametrosKNN, folds, X_train, X_test, y_train, y_test)
    melhoresResultados["KNN"] = [score, matrix, best_parameters]

    print("\n==== Executando --> SVM ====")
    score, matrix, best_parameters = supportVectorMachine(parametersSVM, folds, X_train, X_test, y_train, y_test)
    melhoresResultados["SVM"] = [score, matrix, best_parameters]

    print("\n==== Executando --> Random Forest ====")
    score, matrix, best_parameters = randomForestClassifier(parametersRandomForest, folds, X_train, X_test, y_train, y_test)
    melhoresResultados["RandomForest"] = [score, matrix, best_parameters]

    print("\n==== Executando --> Bagging ====")
    score, matrix, best_parameters = baggingClassifier(parametersBagging, folds, X_train, X_test, y_train, y_test)
    melhoresResultados["Bagging"] = [score, matrix, best_parameters]

    print("\n==== Executando --> MLP_Classifier ====")
    score, matrix, best_parameters = mlpClassifier(parametersMLP, folds, X_train, X_test, y_train, y_test)
    melhoresResultados["MLP_Classifier"] = [score, matrix, best_parameters]

    print("================================")
    print("\nResultados encontrados: \n")
    melhorAcuracia = ["", 0]
    for key in melhoresResultados:
        print("Classification accuracy {}: {}".format(
            key, melhoresResultados[key][0]))
        if melhorAcuracia[1] < melhoresResultados[key][0]:
            melhorAcuracia[0] = key
            melhorAcuracia[1] = melhoresResultados[key][0]

    print("================================")
    print("\nMelhor resultado foi do: {} \nAcuracia de: {}".format(
        melhorAcuracia[0], melhorAcuracia[1]))
    for key in melhoresResultados:
        if key == melhorAcuracia[0]:
            score, matrix, best_parameters = melhoresResultados[key][
                0], melhoresResultados[key][1], melhoresResultados[key][2]

    print("\nParametros utilizados: {}".format(best_parameters))
    plotResultados(matrix)
    
    ratings_matrix = dataset_main.pivot_table(values='Rating', index='IdUser', columns='IdProduct', fill_value=0)
    print(f"\nRating Matrix: \n{ratings_matrix.head()}")
    
    X_matrix = ratings_matrix.T
    
    # Decompondo a Matrix
    SVD = TruncatedSVD(n_components=10)
    decomposed_matrix = SVD.fit_transform(X_matrix)
    print("\nDecompondo a Matrix: ", decomposed_matrix.shape)
    
    # Matriz de correlação
    correlation_matrix = np.corrcoef(decomposed_matrix)
    print("\nMatrix de correlação: ", correlation_matrix.shape)
    
    # Número do índice do ID do produto adquirido pelo cliente
    try:
        id_produto = "B00000K135"
        product_names = list(X_matrix.index)
        product_ID = product_names.index(id_produto)
        
        # Correlação de todos os itens com o item comprado por este cliente com base em itens avaliados por outros clientes que compraram o mesmo produto
        correlation_product_ID = correlation_matrix[product_ID]
        
        # Recomendar os 25 principais produtos altamente correlacionados em sequência
        Recommend = list(X_matrix.index[correlation_product_ID > 0.65])

        # Remove o item já comprado pelo cliente
        Recommend.remove(id_produto) 

        print(f"\nItens que foram recomendados: \n{Recommend[0:24]}")
        
    except:
        print("O Id do cliente inseredo não está na base de dados!")

