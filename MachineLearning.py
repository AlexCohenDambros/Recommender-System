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

dataset = pd.read_csv("ratings_Electronics.csv", names=[
                      "IdUser", "IdProduct", "Rating", "TimesTamp"], sep=",")


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


def machineLearning(dataset):
    melhoresResultados = {}
    print("="*50)

    dataset = dataset.head(1000)
    
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


machineLearning(dataset)
