import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, kernel='linear', sigma=1.0, degree=3):
        """
        Support Vector Machine (SVM) Classifier from Scratch.

        Parameters:
        -----------
        learning_rate : float
            Learning rate for optimization (e.g., in gradient descent).
        lambda_param : float
            Regularization parameter (C = 1/lambda). Controls the trade-off between margin maximization and error.
        n_iters : int
            Number of iterations / epochs for training.
        kernel : str
            Type of kernel to use: 'linear', 'rbf' (Gaussian), or 'poly' (Polynomial).
        sigma : float
            Parameter for the RBF kernel.
        degree : int
            Parameter for the Polynomial kernel.
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel_type = kernel
        self.sigma = sigma
        self.degree = degree
        
        # Model weights and bias (for linear kernel in primal form)
        self.w = None
        self.b = None
        
        # For dual formulation and kernel SVM:
        # We need to store support vectors, their labels, and their Lagrange multipliers (alpha)
        self.X_train = None
        self.y_train = None
        self.alphas = None

    def _kernel(self, x1, x2):
        """
        Computes the kernel function between two vectors or matrices.
        
        TODO: Implement the kernel mathematical functions.
        - Linear: K(x1, x2) = x1 . x2
        - RBF (Radial Basis Function): K(x1, x2) = exp(-||x1 - x2||^2 / (2 * sigma^2))
        - Polynomial: K(x1, x2) = (x1 . x2 + 1)^degree
        """
        if self.kernel_type == 'linear':
            return np.dot(x1, x2)
        elif self.kernel_type == 'rbf':
            # RBF Kernel implementation
            pass
        elif self.kernel_type == 'poly':
            # Polynomial Kernel implementation
            pass
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def fit(self, X, y):
        """
        Train the SVM model on the dataset X with labels y.
        
        Note:
        - Labels y should be binary and encoded as -1 and 1 (instead of 0 and 1).
        
        TODO: Implement the training algorithm.
        You can choose either:
        1. Pegasos or Standard Gradient Descent (for primal soft-margin SVM - works best with linear kernel).
        2. SMO (Sequential Minimal Optimization) or Quadratic Programming Solver (for dual SVM - required for kernel trick).
        """
        n_samples, n_features = X.shape
        
        # Ensure labels are -1 and 1
        y_ = np.where(y <= 0, -1, 1)
        
        # Placeholder initialization for gradient descent (primal form)
        self.w = np.zeros(n_features)
        self.b = 0

        # TODO: Implement the optimization loop here
        # Example for Primal SVM via Gradient Descent:
        # for epoch in range(self.n_iters):
        #     for idx, x_i in enumerate(X):
        #         condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
        #         if condition:
        #             self.w -= self.lr * (2 * self.lambda_param * self.w)
        #         else:
        #             self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
        #             self.b -= self.lr * y_[idx]
        
        pass

    def predict(self, X):
        """
        Predict the class labels for the input data X.
        
        TODO: Implement the prediction formula.
        - For linear SVM in primal form: sign(w . X - b)
        - For Kernel SVM in dual form: sign(sum(alpha_i * y_i * K(x_i, x) - b))
        """
        # Placeholder implementation (returning zeros)
        return np.zeros(X.shape[0])
