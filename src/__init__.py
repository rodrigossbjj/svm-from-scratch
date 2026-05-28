# SVM from Scratch Package

from .svm import SVM
from .data_generator import generate_linearly_separable, generate_circular_dataset, generate_moons_dataset
from .utils import plot_decision_boundary, calculate_accuracy

__all__ = [
    'SVM',
    'generate_linearly_separable',
    'generate_circular_dataset',
    'generate_moons_dataset',
    'plot_decision_boundary',
    'calculate_accuracy'
]
