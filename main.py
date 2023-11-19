import pandas as pd
import os
import time
from tabulate import tabulate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

path = "./data/iris.data"
quantidade_teste = 100


def load_data(path):
    diretorio = os.path.join(os.path.dirname(__file__), path)

    nomes = ["SEPAL_LENGHT", "SEPAL_WIDTH", "PETAL_LENGHT", "PETAL_WIDTH", "CLASS"]

    return pd.read_table(
        filepath_or_buffer=diretorio, header=None, names=nomes, sep=","
    )


def classificar(i_tr, i_te, o_tr, o_te, metodos):
    inicio_treino = time.time()
    metodos[1].fit(i_tr, o_tr)
    fim_treino = time.time()
    tempo_treino = fim_treino - inicio_treino
    predicao = metodos[1].predict(i_te)
    fim_predicao = time.time()
    tempo_predicao = fim_predicao - fim_treino
    acuracia = accuracy_score(o_te, predicao)
    precisao = precision_score(o_te, predicao, average="macro", zero_division=0)
    recall = recall_score(o_te, predicao, average="macro")
    f1 = f1_score(o_te, predicao, average="macro")
    return [acuracia, precisao, recall, f1, tempo_treino, tempo_predicao]


def benchmark(metodos, proporcao):
    dados = load_data(path)
    input = dados.drop("CLASS", axis=1)
    output = dados.CLASS
    input_treino, input_teste, output_treino, output_teste = train_test_split(
        input, output, train_size=proporcao
    )

    resultado = {}

    for met in metodos:
        bench = classificar(input_treino, input_teste, output_treino, output_teste, met)
        resultado[met[0]] = bench

    return resultado


if __name__ == "__main__":
    metodos = [
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("SVC", SVC()),
        ("Arvore", DecisionTreeClassifier()),
        (
            "MLP",
            MLPClassifier(
                solver="lbfgs",
                alpha=1e-5,
                hidden_layer_sizes=(5, 2),
                random_state=1,
                max_iter=1000,
            ),
        ),
        ("Naive Bayes", GaussianNB()),
    ]

    final = {}

    for i in range(quantidade_teste):
        res = benchmark(metodos, 2 / 3)
        for chave in res:
            temp = list(map(lambda x: x / quantidade_teste, res[chave]))
            f = final.get(chave, [0, 0, 0, 0, 0, 0])
            final[chave] = list(map(lambda x: x[0] + x[1], zip(temp, f)))

    tabela = []
    for chave, valores in final.items():
        tabela.append([chave] + valores)

    headers = [
        "Nome método",
        "Acurácia",
        "Precisao",
        "recall",
        "F1",
        "Tempo Treino",
        "Tempo Predição",
    ]
    print(f"Classifcador executado {quantidade_teste} vezes")
    print(tabulate(tabela, headers=headers, tablefmt="grid"))
