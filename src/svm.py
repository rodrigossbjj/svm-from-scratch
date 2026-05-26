import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, kernel='linear', sigma=1.0, degree=3):
        """
        Classificador de Máquina de Vetores de Suporte (SVM) implementado do zero.

        Parâmetros:
        -----------
        learning_rate : float
            Taxa de aprendizado para a otimização (usada no SMO simplificado e atualização).
        lambda_param : float
            Parâmetro de regularização (C = 1/lambda). Controla o compromisso entre a
            maximização da margem e a minimização do erro de treinamento.
        n_iters : int
            Número máximo de iterações/épocas de treinamento.
        kernel : str
            Tipo de função de kernel a ser usada: 'linear', 'rbf' (Gaussiano) ou 'poly' (Polinomial).
        sigma : float
            Parâmetro para o kernel RBF (largura da banda de similaridade).
        degree : int
            Grau para o kernel Polinomial.
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel_type = kernel
        self.sigma = sigma
        self.degree = degree
        
        # Vetor de pesos e viés para a formulação linear primal (opcional/compatibilidade)
        self.w = None
        self.b = 0.0
        
        # Atributos para formulação dual (necessários para o Truque do Kernel)
        self.X_train = None
        self.y_train = None
        self.alphas = None
        
        # Vetores de suporte filtrados (para predição rápida pós-treinamento)
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None

    def _kernel(self, x1, x2):
        """
        Calcula a função de kernel entre dois vetores ou duas matrizes de dados.
        Suporta cálculo vetor-vetor, matriz-vetor e matriz-matriz.
        """
        if self.kernel_type == 'linear':
            # Kernel Linear: K(x1, x2) = x1 . x2^T
            return np.dot(x1, x2.T)
            
        elif self.kernel_type == 'rbf':
            # Kernel RBF (Gaussiano): K(x1, x2) = exp(-||x1 - x2||^2 / (2 * sigma^2))
            if x1.ndim == 1 and x2.ndim == 1:
                sq_dist = np.sum((x1 - x2) ** 2)
            elif x1.ndim > 1 and x2.ndim == 1:
                # Distância de múltiplos pontos (matriz x1) para um ponto (vetor x2)
                sq_dist = np.sum((x1 - x2) ** 2, axis=-1)
            elif x1.ndim == 1 and x2.ndim > 1:
                # Distância de um ponto (vetor x1) para múltiplos pontos (matriz x2)
                sq_dist = np.sum((x1 - x2) ** 2, axis=-1)
            else:
                # Distância par a par entre duas matrizes (M x D) e (N x D)
                # Usamos a identidade algébrica: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a . b^T
                sq_dist = (np.sum(x1 ** 2, axis=1, keepdims=True) + 
                           np.sum(x2 ** 2, axis=1) - 
                           2 * np.dot(x1, x2.T))
            return np.exp(-sq_dist / (2 * (self.sigma ** 2)))
            
        elif self.kernel_type == 'poly':
            # Kernel Polinomial: K(x1, x2) = (x1 . x2^T + 1)^degree
            return (np.dot(x1, x2.T) + 1) ** self.degree
            
        else:
            raise ValueError(f"Tipo de kernel desconhecido: {self.kernel_type}")

    def fit(self, X, y):
        """
        Treina o modelo SVM usando o algoritmo SMO (Sequential Minimal Optimization) Simplificado.

        Parâmetros:
        -----------
        X : numpy.ndarray
            Matriz de dados de treino de formato (n_samples, n_features).
        y : numpy.ndarray
            Rótulos correspondentes de formato (n_samples,). Devem ser codificados como -1 ou 1.
        """
        n_samples, n_features = X.shape
        
        # Garante que os rótulos de classe sejam exatamente -1 e 1
        y_ = np.where(y <= 0, -1, 1)
        
        # Converte lambda_param (regularização L2) para o parâmetro C do SVM clássico
        # C controla o peso dos erros de treinamento (C grande = margem rígida, C pequeno = margem suave)
        C = 1.0 / self.lambda_param if self.lambda_param > 0 else 1e9
        
        # Inicialização dos multiplicadores de Lagrange (alphas) e do viés (bias)
        self.alphas = np.zeros(n_samples)
        self.b = 0.0
        
        # Armazena os dados de treino para avaliação do Kernel
        self.X_train = X
        self.y_train = y_
        
        # Pré-computa a matriz gramiana de Kernel (Gram Matrix) para velocidade durante o treino
        K = self._kernel(X, X)
        
        # Tolerância numérica para violações das condições KKT (Karush-Kuhn-Tucker)
        tol = 1e-3
        
        # Número máximo de épocas sem nenhuma atualização nos alphas antes de parar
        max_passes = 10
        passes = 0
        
        # Loop de otimização iterativa (SMO Simplificado)
        for epoch in range(self.n_iters):
            num_changed_alphas = 0
            
            for i in range(n_samples):
                # 1. Calcula a predição atual f(x_i) e o erro E_i para o exemplo i
                # f(x_i) = sum_k (alpha_k * y_k * K(x_k, x_i)) - b
                f_i = np.sum(self.alphas * self.y_train * K[:, i]) - self.b
                E_i = f_i - self.y_train[i]
                
                # 2. Verifica se o exemplo viola as condições KKT com base na tolerância
                # Violações acontecem se:
                # - y_i * E_i < -tol e alpha_i < C (deveria ser maior, mas está limitado por C)
                # - y_i * E_i > tol e alpha_i > 0 (deveria ser menor, mas está limitado por 0)
                if ((self.y_train[i] * E_i < -tol and self.alphas[i] < C) or 
                    (self.y_train[i] * E_i > tol and self.alphas[i] > 0)):
                    
                    # 3. Seleciona aleatoriamente um segundo exemplo j (j != i) para otimizar em par
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    
                    # 4. Calcula a predição f(x_j) e o erro E_j
                    f_j = np.sum(self.alphas * self.y_train * K[:, j]) - self.b
                    E_j = f_j - self.y_train[j]
                    
                    # Salva os valores antigos de alpha_i e alpha_j
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    
                    # 5. Calcula os limites inferior (L) e superior (H) para garantir que alphas permaneçam viáveis
                    # de acordo com o intervalo [0, C] e a restrição de igualdade sum(alpha * y) = 0.
                    if self.y_train[i] != self.y_train[j]:
                        L = max(0.0, self.alphas[j] - self.alphas[i])
                        H = min(C, C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0.0, self.alphas[i] + self.alphas[j] - C)
                        H = min(C, self.alphas[i] + self.alphas[j])
                        
                    if L == H:
                        continue
                        
                    # 6. Calcula a derivada de segunda ordem da função objetivo (eta)
                    # eta = 2 * K(x_i, x_j) - K(x_i, x_i) - K(x_j, x_j)
                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    
                    # Se eta for maior ou igual a zero, a função objetivo não é estritamente côncava nesse par
                    if eta >= 0:
                        continue
                        
                    # 7. Atualiza o valor de alpha_j ao longo da direção de máxima subida
                    self.alphas[j] = self.alphas[j] - (self.y_train[j] * (E_i - E_j)) / eta
                    
                    # 8. Restringe (clip) o valor de alpha_j dentro das fronteiras viáveis [L, H]
                    if self.alphas[j] > H:
                        self.alphas[j] = H
                    elif self.alphas[j] < L:
                        self.alphas[j] = L
                        
                    # Se o passo de mudança em alpha_j for desprezível, pula para o próximo
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                        
                    # 9. Atualiza alpha_i na direção oposta para manter a restrição:
                    # alpha_i * y_i + alpha_j * y_j = constante
                    self.alphas[i] = self.alphas[i] + self.y_train[i] * self.y_train[j] * (alpha_j_old - self.alphas[j])
                    
                    # 10. Computa os possíveis novos valores para o viés b1 (com base em i) e b2 (com base em j)
                    b1 = (self.b + E_i + 
                          self.y_train[i] * (self.alphas[i] - alpha_i_old) * K[i, i] + 
                          self.y_train[j] * (self.alphas[j] - alpha_j_old) * K[i, j])
                          
                    b2 = (self.b + E_j + 
                          self.y_train[i] * (self.alphas[i] - alpha_i_old) * K[i, j] + 
                          self.y_train[j] * (self.alphas[j] - alpha_j_old) * K[j, j])
                    
                    # Escolhe o novo viés com base em se os alphas otimizados estão nas fronteiras
                    if 0 < self.alphas[i] < C:
                        self.b = b1
                    elif 0 < self.alphas[j] < C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0
                        
                    num_changed_alphas += 1
            
            # Controle de convergência precoce
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
                
            if passes >= max_passes:
                break
                
        # Pós-processamento: Filtra e salva apenas os vetores de suporte (onde alpha > 0)
        # para otimizar drasticamente a velocidade de predição.
        sv_mask = self.alphas > 1e-5
        self.support_vectors = self.X_train[sv_mask]
        self.support_vector_labels = self.y_train[sv_mask]
        self.support_vector_alphas = self.alphas[sv_mask]
        
        # Computa w explicitamente caso o kernel seja linear (para compatibilidade primal)
        # w = sum (alpha_i * y_i * x_i)
        if self.kernel_type == 'linear':
            self.w = np.dot(self.alphas * self.y_train, self.X_train)
        else:
            self.w = None

    def decision_function(self, X):
        """
        Calcula a pontuação de decisão (distância com sinal ao hiperplano separador) para X.

        Parâmetros:
        -----------
        X : numpy.ndarray
            Matriz de dados de entrada de formato (n_samples, n_features) ou vetor 1D.

        Retorna:
        --------
        scores : numpy.ndarray ou float
            A pontuação de decisão para cada ponto (distâncias com sinal ao hiperplano).
        """
        # Se o kernel for linear e os pesos w estiverem disponíveis, calculamos no formato primal rápido
        if self.kernel_type == 'linear' and self.w is not None:
            return np.dot(X, self.w) - self.b
        else:
            # Caso contrário, usamos a predição dual com os vetores de suporte salvos
            if self.support_vectors is None or len(self.support_vector_alphas) == 0:
                # Caso o modelo não tenha vetores de suporte (não treinado), retorna o viés negativo
                return -self.b * np.ones(X.shape[0]) if X.ndim > 1 else -self.b
                
            # Calcula a similaridade (kernel) entre os vetores de suporte e os dados de teste X
            K_test = self._kernel(self.support_vectors, X)
            
            # f(x) = sum (alpha_i * y_i * K(x_i, x)) - b
            return np.dot(self.support_vector_alphas * self.support_vector_labels, K_test) - self.b

    def predict(self, X):
        """
        Classifica as amostras em X nas classes -1 ou 1 com base no hiperplano de decisão.

        Parâmetros:
        -----------
        X : numpy.ndarray
            Matriz de dados de entrada de formato (n_samples, n_features).

        Retorna:
        --------
        y_pred : numpy.ndarray
            Rótulos preditos (-1 ou 1).
        """
        scores = self.decision_function(X)
        # Retorna 1 se a pontuação for >= 0, caso contrário retorna -1
        if isinstance(scores, np.ndarray):
            return np.where(scores >= 0, 1, -1)
        else:
            return 1 if scores >= 0 else -1
