# SVM do Zero (SVM from Scratch)

Este repositório contém a estrutura inicial para a implementação de um classificador **Support Vector Machine (SVM)** do zero em Python, utilizando conjuntos de dados (datasets) sintéticos para reconhecimento de padrões.

## 📂 Estrutura do Projeto

```text
svm-from-scratch/
│
├── src/
│   ├── __init__.py          # Define o diretório como pacote Python
│   ├── svm.py               # Esqueleto da implementação do SVM (Matemática e Ajuste)
│   ├── data_generator.py    # Gerador de dados sintéticos (Linear, Circular, Moons)
│   └── utils.py             # Funções de suporte (visualização gráfica das margens e acurácia)
│
├── main.py                  # Script principal para rodar o pipeline completo
├── requirements.txt         # Dependências do projeto (numpy, matplotlib, scikit-learn)
└── README.md                # Instruções e conceitos matemáticos
```

---

## 🛠️ Como Iniciar

1. **Criar e ativar um ambiente virtual (opcional, mas recomendado):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Instalar as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Executar o código:**
   ```bash
   python main.py
   ```

---

## 📚 Conceitos Teóricos para Implementação

Para construir o SVM do zero, você precisará implementar os conceitos matemáticos a seguir:

### 1. A Equação do Hiperplano
O SVM tenta encontrar um hiperplano separador com a maior margem possível. A equação de predição é dada por:

$$f(x) = w \cdot x - b$$

Onde:
*   $w$ é o vetor de pesos (perpendicular ao hiperplano).
*   $b$ é o viés (bias).
*   Se $f(x) \geq 1$, a classe predita é $+1$.
*   Se $f(x) \leq -1$, a classe predita é $-1$.

A classe predita final é $\text{sign}(w \cdot x - b)$.

### 2. Margem Suave e Função de Perda (Hinge Loss)
Em casos reais, os dados podem não ser perfeitamente separáveis. Usamos a formulação de **Margem Suave** (Soft-Margin) com a perda **Hinge Loss**:

$$L = \lambda \|w\|^2 + \frac{1}{N} \sum_{i=1}^{N} \max(0, 1 - y_i(w \cdot x_i - b))$$

Onde:
*   $\lambda$ (lambda) é o termo de regularização que controla a largura da margem.
*   $y_i \in \{-1, 1\}$ é a classe real do exemplo $i$.

### 3. Otimização via Gradiente Descendente
Para atualizar $w$ e $b$ a cada passo de treinamento com um exemplo $(x_i, y_i)$:

*   **Caso correto e fora da margem** ($y_i(w \cdot x_i - b) \geq 1$):
    $$w \leftarrow w - \eta \cdot (2 \lambda w)$$
    $$b \leftarrow b$$

*   **Caso incorreto ou dentro da margem** ($y_i(w \cdot x_i - b) < 1$):
    $$w \leftarrow w - \eta \cdot (2 \lambda w - y_i x_i)$$
    $$b \leftarrow b - \eta \cdot y_i$$

*(Onde $\eta$ é a taxa de aprendizado `learning_rate`)*

### 4. Truque do Kernel (Kernel Trick)
Para datasets não-lineares (como o circular ou em formato de luas), usamos funções de Kernel para projetar os dados em uma dimensão superior onde eles são linearmente separáveis:

*   **Linear:** $K(x_1, x_2) = x_1 \cdot x_2$
*   **Polinomial:** $K(x_1, x_2) = (x_1 \cdot x_2 + 1)^d$
*   **RBF (Gaussiano):** $K(x_1, x_2) = \exp\left(-\frac{\|x_1 - x_2\|^2}{2\sigma^2}\right)$
