import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y, title="Fronteira de Decisão do SVM", save_path=None):
    """
    Plota os pontos do dataset sintético e desenha a fronteira de decisão (hiperplano) e as margens do SVM.
    Suporta tanto kernels lineares quanto não-lineares e destaca os vetores de suporte encontrados.
    
    Parâmetros:
    -----------
    model : SVM
        O modelo SVM já treinado.
    X : numpy.ndarray
        Matriz de características/features de entrada de formato (n_samples, 2).
    y : numpy.ndarray
        Rótulos reais (-1 ou 1) correspondentes aos dados de formato (n_samples,).
    title : str
        Título do gráfico.
    save_path : str (opcional)
        Caminho de arquivo para salvar o gráfico gerado (ex: 'fronteira.png'). Se omitido, plota na tela.
    """
    plt.figure(figsize=(10, 8))
    
    # 1. Plota os pontos das duas classes do dataset com cores distintas (Azul e Vermelho)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', label='Classe +1', edgecolors='k', s=50)
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', marker='x', label='Classe -1', s=50)
    
    # 2. Destaca os Vetores de Suporte com um círculo verde ao redor deles
    if hasattr(model, 'support_vectors') and model.support_vectors is not None and len(model.support_vectors) > 0:
        plt.scatter(model.support_vectors[:, 0], model.support_vectors[:, 1], 
                    s=150, facecolors='none', edgecolors='green', linewidths=1.5, 
                    label='Vetores de Suporte')
    
    # 3. Cria uma grade retangular (grid) cobrindo todo o plano dos dados
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Gera 100 pontos lineares em cada eixo para criar a malha
    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    
    # 4. Avalia a função de decisão do modelo em cada ponto da grade criada
    try:
        if hasattr(model, 'decision_function'):
            Z = model.decision_function(xy).reshape(XX.shape)
        else:
            # Fallback seguro calculando o produto linear se w estiver definido
            if model.w is not None:
                Z = (np.dot(xy, model.w) - model.b).reshape(XX.shape)
            else:
                Z = np.zeros(XX.shape)
                
        # 5. Desenha as curvas de nível (contours):
        # - Nível 0: A fronteira de decisão principal (hiperplano separador)
        # - Níveis -1 e 1: As margens de suporte
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    except Exception as e:
        print(f"Erro ao plotar contorno: {e}. Certifique-se de que o modelo foi treinado com sucesso.")
        
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Característica 1 (Feature 1)", fontsize=11)
    plt.ylabel("Característica 2 (Feature 2)", fontsize=11)
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 6. Salva o gráfico ou renderiza-o interativamente
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"-> Gráfico salvo com sucesso em: {save_path}")
        plt.close()
    else:
        plt.show()

def calculate_accuracy(y_true, y_pred):
    """
    Calcula a acurácia percentual das previsões.
    
    Parâmetros:
    -----------
    y_true : numpy.ndarray
        Rótulos reais.
    y_pred : numpy.ndarray
        Rótulos preditos pelo classificador.
        
    Retorna:
    --------
    accuracy : float
        Percentual de acerto (de 0.0 a 100.0).
    """
    return np.mean(y_true == y_pred) * 100
