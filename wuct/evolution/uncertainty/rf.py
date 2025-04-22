import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

from .base import UncertaintyEstimator
from .factory import UncertaintyEstimatorFactory

@UncertaintyEstimatorFactory.register("rf")
class RFUncertaintyEstimator(UncertaintyEstimator):
    """Random Forest based uncertainty estimator for time series forecasts.
    
    This class implements uncertainty estimation using Random Forest models
    by generating multiple predictions through random tree sampling.

    Example:
        >>> # Create and train estimator
        >>> estimator = UncertaintyEstimatorFactory.create("rf", n_samples=1000)
        >>> estimator.fit(training_data)
        >>> 
        >>> # Generate samples for new data
        >>> samples = estimator.generate_samples(test_data)
        >>> 
        >>> # Save and load model
        >>> estimator.save_model("rf_model.joblib")
        >>> loaded_estimator = RFUncertaintyEstimator.load_model("rf_model.joblib")
    
    Args:
        n_samples (int): Number of samples to generate for uncertainty estimation
        n_estimators (int): Number of trees in the Random Forest
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    """
    
    def __init__(self, n_samples: int = 1000, n_estimators: int = 100, 
                 test_size: float = 0.2, random_state: int = 42):
        super().__init__(n_samples)
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        
    def fit(self, data: np.ndarray, **kwargs) -> 'RFUncertaintyEstimator':
        """
        Train the Random Forest model on the input data.
        
        Args:
            data: Input data of shape (n_samples, n_features + 1)
                 Last column is assumed to be the target variable
            **kwargs: Additional training parameters
        
        Returns:
            self: The fitted estimator
        """
        # Split features and target
        X = data[:, :-1]
        y = data[:, -1]
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # Initialize and train the model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        
        return self
    
    def generate_samples(self, data: np.ndarray, n_trees: int = 30, **kwargs) -> np.ndarray:
        """
        Generate samples using random subsets of trees from the Random Forest.
        
        Args:
            data: Input data of shape (n_samples, n_features)
            n_trees: Number of trees to use for each sample
            **kwargs: Additional generation parameters
            
        Returns:
            np.ndarray: Generated samples of shape (n_samples, n_samples_per_point)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        error_samples = np.zeros((data.shape[0], self.n_samples))
        
        for i in range(self.n_samples):
            # Randomly select a subset of trees for each prediction
            random_trees = np.random.choice(
                self.model.estimators_, 
                size=n_trees, 
                replace=False
            )
            predictions = np.array([tree.predict(data) for tree in random_trees])
            error_samples[:, i] = np.mean(predictions, axis=0)
            
        return error_samples
    
    def save_model(self, save_path: str) -> None:
        """
        Save the fitted Random Forest model.
        
        Args:
            save_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not fitted")
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.model, save_path)
    
    @classmethod
    def load_model(cls, load_path: str) -> 'RFUncertaintyEstimator':
        """
        Load a saved Random Forest model.
        
        Args:
            load_path: Path to load the model from
            
        Returns:
            RFUncertaintyEstimator: Loaded model instance
        """
        instance = cls()
        instance.model = joblib.load(load_path)
        return instance
    
    def _combine_prediction_and_error(self, predictions: np.ndarray, 
                                    error_samples: np.ndarray) -> np.ndarray:
        """
        Combine predictions with error samples.
        
        Args:
            predictions: Base predictions of shape (n_samples,)
            error_samples: Error samples of shape (n_samples, n_samples_per_point)
            
        Returns:
            np.ndarray: Combined predictions of shape (n_samples, n_samples_per_point)
        """
        return predictions[:, np.newaxis] - error_samples