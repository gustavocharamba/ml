import os
import numpy as np
import cv2
from tqdm import tqdm

def carregar_imagens(caminho_base, tamanho=(512, 512)):
    categorias = ['tubarao', 'mar', 'naoSei']
    dados = []
    rotulos = []

    for idx, categoria in enumerate(categorias):
        caminho = os.path.join(caminho_base, categoria)
        if not os.path.exists(caminho):
            print(f"Aviso: Pasta {caminho} não encontrada.")
            continue

        arquivos = os.listdir(caminho)
        for arquivo in arquivos:
            img_path = os.path.join(caminho, arquivo)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Erro ao carregar: {img_path}")
                continue
            img = cv2.resize(img, tamanho)  # Garantindo tamanho correto
            img = img / 255.0  # Normalizando
            dados.append(img.flatten())  # Convertendo para vetor
            rotulos.append(idx)

    dados = np.array(dados)
    rotulos = np.array(rotulos)
    return dados, rotulos

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def reg_l2(w, lamb_l2):
    return lamb_l2 * w

def pesos(entrada, escondida, saida):
    escondida = int(escondida)
    saida = int(saida)  # Add this line to convert saida to int
    w1 = np.random.randn(entrada, escondida) * np.sqrt(2.0 / (entrada + escondida))
    b1 = np.zeros((1, escondida))
    w2 = np.random.randn(escondida, saida) * np.sqrt(2.0 / (escondida + saida))
    b2 = np.zeros((1, saida))
    return w1, b1, w2, b2

def forward(x, w1, b1, w2, b2):
    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    saida = softmax(z2)
    return saida, a1, z1, z2

def backward(x, y, saida, a1, z1, w1, b1, w2, b2, lamb_l2):
    m = x.shape[0]
    erro_saida = saida - y
    dw2 = np.dot(a1.T, erro_saida) / m + reg_l2(w2, lamb_l2)
    db2 = np.sum(erro_saida, axis=0, keepdims=True) / m
    erro_escondida = np.dot(erro_saida, w2.T) * sigmoid_deriv(a1)
    dw1 = np.dot(x.T, erro_escondida) / m + reg_l2(w1, lamb_l2)
    db1 = np.sum(erro_escondida, axis=0, keepdims=True) / m
    return dw1, db1, dw2, db2

def treinamento(x, y, epocas=10, taxa=0.01, k_folds=5, lamb_l2=0.01):
    y_onehot = np.zeros((y.shape[0], len(np.unique(y))))
    for i, classe in enumerate(np.unique(y)):
        y_onehot[:, i] = (y == classe).astype(int)

    entrada = x.shape[1]
    escondida = 15
    saida = y_onehot.shape[1]  # Corrigido para o número de classes

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    tamanho = len(indices) // k_folds

    acuracias = []
    for j in range(k_folds):
        inicio, fim = j * tamanho, (j + 1) * tamanho
        indice_valid = indices[inicio:fim]
        indices_treino = np.concatenate([indices[:inicio], indices[fim:]])
        x_treino, y_treino = x[indices_treino], y_onehot[indices_treino]
        x_valid, y_valid = x[indice_valid], y_onehot[indice_valid]

        w1, b1, w2, b2 = pesos(entrada, escondida, saida)

        for epoca in tqdm(range(epocas), desc="Treinando", unit="época"):
            permut = np.random.permutation(len(x_treino))
            x_shuffle, y_shuffle = x_treino[permut], y_treino[permut]
            batch = 32

            for i in range(0, len(x_shuffle), batch):
                x_batch, y_batch = x_shuffle[i:i + batch], y_shuffle[i:i + batch]
                saida, a1, z1, z2 = forward(x_batch, w1, b1, w2, b2)
                dw1, db1, dw2, db2 = backward(x_batch, y_batch, saida, a1, z1, w1, b1, w2, b2, lamb_l2)
                w1 -= taxa * dw1
                b1 -= taxa * db1
                w2 -= taxa * dw2
                b2 -= taxa * db2

        saida_valid, _, _, _ = forward(x_valid, w1, b1, w2, b2)
        acuracia = np.mean(np.argmax(saida_valid, axis=1) == np.argmax(y_valid, axis=1))
        acuracias.append(acuracia)

    media = np.mean(acuracias)
    desvio = np.std(acuracias)
    print(f"Acurácia média: {media:.4f}")
    print(f"Desvio padrão: {desvio:.4f}")
    return media, desvio

# Exemplo de uso:
caminho_base = ""  # Substitua com o caminho correto
x, y = carregar_imagens(caminho_base)
print(f"Total de amostras carregadas: {len(y)}")

# Executando o treinamento e calculando a acurácia
media_acuracia, desvio_acuracia = treinamento(x, y)
