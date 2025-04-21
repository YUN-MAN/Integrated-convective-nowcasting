# Abstract uncertainty estimator

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Union
import numpy as np

class RunMode(Enum):
    TRAIN = "train"
    PREDICT = "predict"

class UncertaintyEstimator(ABC):
    """Abstract base class for uncertainty estimation."""
    
    def __init__(self, n_samples: int = 1000):
        """
        Initialize the uncertainty estimator.
        
        Args:
            n_samples: Number of samples to generate for uncertainty estimation
        """
        self.n_samples = n_samples
    
    @abstractmethod
    def fit(self, data: np.ndarray, **kwargs) -> 'UncertaintyEstimator':
        """
        Fit the uncertainty estimator to training data.
        
        Args:
            data: Training data
            **kwargs: Additional fitting parameters
            
        Returns:
            self: The fitted estimator
        """
        pass
    
    @abstractmethod
    def generate_samples(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate uncertainty samples.
        
        Args:
            data: Input data for generating samples
            **kwargs: Additional generation parameters
            
        Returns:
            np.ndarray: Generated samples
        """
        pass
    
    @abstractmethod
    def save_model(self, save_path: str) -> None:
        """
        Save the fitted model parameters.
        
        Args:
            save_path: Path to save the model
        """
        pass
    
    @classmethod
    @abstractmethod
    def load_model(cls, load_path: str) -> 'UncertaintyEstimator':
        """
        Load a saved model.
        
        Args:
            load_path: Path to load the model from
            
        Returns:
            UncertaintyEstimator: Loaded model
        """
        pass