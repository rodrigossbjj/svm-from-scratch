import numpy as np
from src.data_generator import generate_linearly_separable, generate_circular_dataset, generate_moons_dataset
from src.svm import SVM
from src.utils import plot_decision_boundary, calculate_accuracy

def main():
    print("=== Support Vector Machine (SVM) from Scratch ===")
    
    # 1. Choose and generate a synthetic dataset
    # options: 'linear', 'circular', 'moons'
    dataset_type = 'linear'
    print(f"\n[1] Generating synthetic dataset of type: '{dataset_type}'...")
    
    if dataset_type == 'linear':
        X, y = generate_linearly_separable(n_samples=100, noise=0.15)
    elif dataset_type == 'circular':
        X, y = generate_circular_dataset(n_samples=150, noise=0.08)
    else:
        X, y = generate_moons_dataset(n_samples=150, noise=0.1)
        
    print(f"Dataset shape: Features = {X.shape}, Labels = {y.shape}")
    print(f"Classes distribution: {np.sum(y == 1)} of class +1, {np.sum(y == -1)} of class -1")

    # 2. Instantiate the SVM model
    print("\n[2] Initializing SVM classifier...")
    # Adjust parameters (learning rate, lambda, iterations, kernel, etc.)
    model = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000, kernel='linear')

    # 3. Train the model
    print("\n[3] Training model...")
    model.fit(X, y)
    
    # 4. Predict and evaluate
    print("\n[4] Making predictions...")
    predictions = model.predict(X)
    
    acc = calculate_accuracy(y, predictions)
    print(f"Training Accuracy: {acc:.2f}%")

    # 5. Visualize decision boundary
    print("\n[5] Visualizing decision boundary...")
    # Define a custom decision_function for drawing the contours in utils.plot_decision_boundary
    # Add a temporary method to our model instance for demonstration if not implemented
    if not hasattr(model, 'decision_function'):
        # For a standard linear SVM, decision function is w . x - b
        model.decision_function = lambda x: np.dot(x, model.w) - model.b
        
    plot_decision_boundary(model, X, y, title=f"SVM Decision Boundary ({dataset_type.capitalize()} Dataset)")

if __name__ == "__main__":
    main()
