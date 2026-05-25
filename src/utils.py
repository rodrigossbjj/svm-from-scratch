import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y, title="SVM Decision Boundary"):
    """
    Plots the dataset points and the decision boundary / margins of the SVM model.
    Works for both linear and kernel-based predictions.
    
    Parameters:
    -----------
    model : SVM
        The trained SVM model.
    X : numpy.ndarray
        The input features (shape: n_samples x 2)
    y : numpy.ndarray
        The binary class labels (-1 or 1)
    """
    plt.figure(figsize=(10, 8))
    
    # Plot dataset points
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', label='Class +1', edgecolors='k', s=50)
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', marker='x', label='Class -1', s=50)
    
    # Create grid to evaluate model predictions across the space
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    
    # Get model decision values for grid points
    # NOTE: You will need to implement a decision_function in your SVM or use predict
    try:
        # If your model has a decision_function returning the raw distance/score:
        # z = model.decision_function(xy).reshape(XX.shape)
        # For primal linear SVM, decision_function is: np.dot(xy, model.w) - model.b
        if hasattr(model, 'decision_function'):
            Z = model.decision_function(xy).reshape(XX.shape)
        else:
            # Fallback using weights directly if linear primal model
            if model.w is not None:
                Z = (np.dot(xy, model.w) - model.b).reshape(XX.shape)
            else:
                Z = np.zeros(XX.shape)
                
        # Plot decision boundary (Z=0) and margins (Z=-1, Z=1)
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    except Exception as e:
        print(f"Could not plot contour: {e}. Make sure model is trained and weights are initialized.")
        
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def calculate_accuracy(y_true, y_pred):
    """
    Computes accuracy percentage.
    """
    return np.mean(y_true == y_pred) * 100
