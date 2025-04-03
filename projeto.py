import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def carregar_imagens(tamanho=(256, 256)):
    classes = ['tubarao', 'mar', 'naoSei']
    imagens = []
    rotulos = []

    for idx, classe in enumerate(classes):
        pasta_classe = classe
        if not os.path.exists(pasta_classe):
            print(f"Aviso: A pasta {pasta_classe} não existe. Pulando.")
            continue

        for arquivo in os.listdir(pasta_classe):
            if arquivo.endswith(('.png', '.jpg', '.jpeg')):
                caminho_arquivo = os.path.join(pasta_classe, arquivo)
                try:
                    img = cv2.imread(caminho_arquivo, cv2.IMREAD_GRAYSCALE)
                    if img.shape[0] != tamanho[0] or img.shape[1] != tamanho[1]:
                        img = cv2.resize(img, tamanho)
                    img_normalizada = img.astype(np.float32) / 255.0
                    img_achatada = img_normalizada.flatten()
                    imagens.append(img_achatada)
                    rotulos.append(idx)
                except Exception as e:
                    print(f"Erro ao processar {arquivo}: {str(e)}")
    return np.array(imagens), np.array(rotulos)


# Classe da Rede Neural
class RedeNeural:
    def __init__(self, tamanho_entrada, tamanho_oculta, tamanho_saida, regularizacao='L2', lambda_reg=0.01,
                 dropout_rate=0.2):
        self.W1 = np.random.randn(tamanho_entrada, tamanho_oculta) * np.sqrt(2.0 / tamanho_entrada)
        self.b1 = np.zeros((1, tamanho_oculta))
        self.W2 = np.random.randn(tamanho_oculta, tamanho_saida) * np.sqrt(2.0 / tamanho_oculta)
        self.b2 = np.zeros((1, tamanho_saida))
        self.regularizacao = regularizacao
        self.lambda_reg = lambda_reg
        self.dropout_rate = dropout_rate
        self.dropout_mask = None

    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X, training=False):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)

        if training:
            self.dropout_mask = (np.random.rand(*self.A1.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            self.A1 *= self.dropout_mask

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def backward(self, X, Y, learning_rate):
        m = X.shape[0]
        dZ2 = self.A2 - Y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        if self.regularizacao == 'L2':
            dW2 += (self.lambda_reg / m) * self.W2

        dZ1 = np.dot(dZ2, self.W2.T) * (self.Z1 > 0)
        if self.dropout_mask is not None:
            dZ1 *= self.dropout_mask

        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        if self.regularizacao == 'L2':
            dW1 += (self.lambda_reg / m) * self.W1

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def predict(self, X):
        return np.argmax(self.forward(X, training=False), axis=1)

    def train(self, X, Y, epochs=50, batch_size=16, learning_rate=0.0001, k_folds=5):
        kf = KFold(n_splits=k_folds, shuffle=True)
        acc_scores = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            for epoch in range(epochs):
                indices = np.random.permutation(X_train.shape[0])
                X_train = X_train[indices]
                Y_train = Y_train[indices]

                for i in range(0, X_train.shape[0], batch_size):
                    X_batch = X_train[i:i + batch_size]
                    Y_batch = np.eye(3)[Y_train[i:i + batch_size]]
                    self.forward(X_batch, training=True)
                    self.backward(X_batch, Y_batch, learning_rate)

            Y_pred = self.predict(X_val)
            acc = np.mean(Y_pred == Y_val)
            acc_scores.append(acc)

        print(f"Acurácia média: {np.mean(acc_scores):.4f}, Desvio padrão: {np.std(acc_scores):.4f}")


# Função para processar imagem e gerar mapa de calor
def processar_imagem(entrada, saida, modelo):
    imagem = cv2.imread(entrada)
    altura, largura = imagem.shape[:2]
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    mapa_probabilidade = np.zeros((altura, largura), dtype=np.float32)

    janela_tamanho = 64
    passo = 32

    for y in range(0, altura - janela_tamanho + 1, passo):
        for x in range(0, largura - janela_tamanho + 1, passo):
            roi = imagem_cinza[y:y + janela_tamanho, x:x + janela_tamanho]
            roi_resized = cv2.resize(roi, (256, 256))
            roi_norm = roi_resized.astype(np.float32) / 255.0
            roi_flat = roi_norm.flatten()
            predicao = modelo.forward(roi_flat, training=False)
            probabilidade_tubarao = predicao[0][0]
            mapa_probabilidade[y:y + janela_tamanho, x:x + janela_tamanho] += probabilidade_tubarao

    if np.max(mapa_probabilidade) > 0:
        mapa_probabilidade = mapa_probabilidade / np.max(mapa_probabilidade)

    mapa_calor_normalizado = (mapa_probabilidade * 255).astype(np.uint8)
    mapa_calor_colorido = cv2.applyColorMap(mapa_calor_normalizado, cv2.COLORMAP_JET)
    alpha = 0.6
    imagem_com_mapa = cv2.addWeighted(imagem, 1 - alpha, mapa_calor_colorido, alpha, 0)

    cv2.imwrite(saida, imagem_com_mapa)
    print(f"Imagem processada salva em: {saida}")


if __name__ == "__main__":
    X, Y = carregar_imagens()
    modelo = RedeNeural(tamanho_entrada=256 * 256, tamanho_oculta=256, tamanho_saida=3)
    modelo.train(X, Y)

    # Processar uma imagem específica após o treinamento
    processar_imagem("frame_0134.png", "saida.png", modelo)