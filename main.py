import numpy as np
from src.data_generator import generate_linearly_separable, generate_circular_dataset, generate_moons_dataset
from src.svm import SVM
from src.utils import plot_decision_boundary, calculate_accuracy

def run_experiment(dataset_type, kernel_type, lambda_param=0.01, sigma=1.0, degree=3, n_iters=1000):
    """
    Executa um pipeline de ponta a ponta (Geração de dados, Treinamento, Predição e Plotagem)
    para uma configuração específica de dataset e kernel.
    """
    print(f"\n--- Iniciando Experimento: Dataset '{dataset_type.upper()}' com Kernel '{kernel_type.upper()}' ---")
    
    # 1. Geração do Dataset Sintético apropriado
    if dataset_type == 'linear':
        X, y = generate_linearly_separable(n_samples=150, noise=0.12, random_state=42)
    elif dataset_type == 'circular':
        X, y = generate_circular_dataset(n_samples=200, noise=0.06, factor=0.5, random_state=42)
    elif dataset_type == 'moons':
        X, y = generate_moons_dataset(n_samples=200, noise=0.08, random_state=42)
    else:
        raise ValueError(f"Dataset inválido: {dataset_type}")
        
    print(f"[1] Dataset gerado com sucesso!")
    print(f"    - Amostras: {X.shape[0]}, Características: {X.shape[1]}")
    print(f"    - Distribuição: {np.sum(y == 1)} na classe +1 | {np.sum(y == -1)} na classe -1")

    # 2. Inicialização do Classificador SVM do zero
    # O C interno é inversamente proporcional ao lambda_param (C = 1 / lambda)
    print(f"[2] Inicializando classificador SVM...")
    model = SVM(
        learning_rate=0.01, 
        lambda_param=lambda_param, 
        n_iters=n_iters, 
        kernel=kernel_type, 
        sigma=sigma, 
        degree=degree
    )

    # 3. Treinamento do Modelo SVM via algoritmo SMO Simplificado
    print(f"[3] Treinando o SVM usando SMO Simplificado (máx {n_iters} épocas)...")
    model.fit(X, y)
    
    # 4. Avaliação e Métricas de Predição
    print(f"[4] Computando predições e acurácia...")
    predictions = model.predict(X)
    acc = calculate_accuracy(y, predictions)
    
    # Conta quantos vetores de suporte foram retidos (onde alphas > 0)
    n_support_vectors = len(model.support_vector_alphas) if model.support_vector_alphas is not None else 0
    print(f"    -> Acurácia obtida no treino: {acc:.2f}%")
    print(f"    -> Quantidade de Vetores de Suporte: {n_support_vectors} (de {X.shape[0]} amostras)")

    # 5. Visualização gráfica e salvamento do arquivo de imagem resultante
    filename = f"{dataset_type}_decision_boundary.png"
    print(f"[5] Desenhando a fronteira de decisão e salvando o gráfico...")
    title_str = f"SVM com Kernel {kernel_type.capitalize()} ({dataset_type.capitalize()} Dataset)"
    plot_decision_boundary(model, X, y, title=title_str, save_path=filename)
    
    return acc

def main():
    print("==========================================================")
    # Título do script principal rodando no terminal
    print("=== Máquina de Vetores de Suporte (SVM) do Zero em Python ===")
    print("==========================================================")
    
    # Executaremos 3 experimentos clássicos para validar nosso modelo SVM do zero:
    
    # Caso 1: Separação Linear Simples (Linear Kernel)
    # Dados linearmente separáveis são perfeitamente resolvidos pelo kernel linear tradicional.
    run_experiment(
        dataset_type='linear', 
        kernel_type='linear', 
        lambda_param=0.01, 
        n_iters=1000
    )
    
    # Caso 2: Separação Circular Não-Linear (RBF Kernel)
    # Círculos concêntricos não são linearmente separáveis no plano 2D, mas são com o Kernel Gaussiano (RBF).
    run_experiment(
        dataset_type='circular', 
        kernel_type='rbf', 
        lambda_param=0.01, 
        sigma=0.5, 
        n_iters=1500
    )
    
    # Caso 3: Separação em Formato de Luas Intercaladas (RBF Kernel)
    # O clássico dataset de duas "meias-luas" requer uma fronteira de decisão curva, resolvida por RBF.
    run_experiment(
        dataset_type='moons', 
        kernel_type='rbf', 
        lambda_param=0.01, 
        sigma=0.5, 
        n_iters=1500
    )
    
    print("\n==========================================================")
    print("=== Todos os experimentos foram concluídos com sucesso! ===")
    print("Os gráficos foram gravados no diretório atual.")
    print("==========================================================")

if __name__ == "__main__":
    main()
