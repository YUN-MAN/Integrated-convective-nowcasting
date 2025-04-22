import json
import os
import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.tsa.api import VAR
from typing import Optional, Dict

from .base import UncertaintyEstimator
from .factory import UncertaintyEstimatorFactory

@UncertaintyEstimatorFactory.register("var")
class VARUncertaintyEstimator(UncertaintyEstimator):
    """Vector Autoregression based uncertainty estimator.
    
    This class implements uncertainty estimation using Vector Autoregression (VAR)
    combined with Student's t-distribution for modeling forecast uncertainties.

    Example:
        >>> # Create and train estimator
        >>> estimator = UncertaintyEstimatorFactory.create("var", n_samples=1000)
        >>> estimator.fit(training_data)
        >>> 
        >>> # Generate samples for new data
        >>> samples = estimator.generate_samples(test_data)
        >>> 
        >>> # Save and load model
        >>> estimator.save_model("var_model.json")
        >>> loaded_estimator = VARUncertaintyEstimator.load_model("var_model.json")
    
    Args:
        n_samples (int): Number of samples to generate for uncertainty estimation
        maxlags (int): Maximum number of lags for VAR model
    """
    
    def __init__(self, n_samples: int = 1000, maxlags: int = 1):
        super().__init__(n_samples)
        self.maxlags = maxlags
        self.var_results = None
        self.t_params = None
    
    def fit(self, data: np.ndarray, train_size: float = 0.8, 
            column_names: Optional[list[str]] = None) -> 'VARUncertaintyEstimator':
        """
        Fit the VAR model and Student-t distributions to the residuals.
        
        Args:
            data: Input data of shape (n_samples, n_variables)
            train_size: Proportion of data to use for training
            column_names: Names for the variables (default: "t+{5|10|15}")
        """
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            if column_names is None:
                column_names = [f"t+{(i+1)*5}" for i in range(data.shape[1])]
            data = pd.DataFrame(data, columns=column_names)
            
        # Split data
        train_size = int(train_size * len(data))
        self.train_data = data.iloc[:train_size]
        self.test_data = data.iloc[train_size:]
        
        # Fit VAR model
        self.var_model = VAR(self.train_data)
        self.var_results = self.var_model.fit(maxlags=self.maxlags)
        
        # Fit Student-t to residuals
        self.t_params = {}
        residuals = self.var_results.resid
        
        for var in residuals.columns:
            # Fit Student-t
            df, loc, scale = t.fit(residuals[var])
            self.t_params[var] = {'df': df, 'loc': loc, 'scale': scale}
        
        return self
    
    def generate_samples(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate probabilistic forecasts using parametric sampling.
        
        Args:
            data: Initial values for forecast, shape (n_steps, n_variables)
            **kwargs: Additional parameters
                
        Returns:
            np.ndarray: Forecast samples of shape (n_steps, n_samples, n_variables)
        """
        steps = len(data)
        n_vars = data.shape[1]
        forecast_samples = np.zeros((steps, self.n_samples, n_vars))
        
        coefs = self.var_results.coefs
        intercept = self.var_results.intercept
        
        for i in range(steps):
            # Deterministic forecast
            if i < self.var_results.k_ar:
                base_forecast = np.dot(data[i:i+1], coefs[0].T) + intercept
                deterministic_forecast = np.tile(base_forecast, (self.n_samples, 1))
            else:
                deterministic_forecast = (
                    np.dot(forecast_samples[i-1, :, :], coefs[0].T) + intercept
                )
            
            # Add sampled residuals
            sampled_residuals = self._generate_residual_samples(self.n_samples, n_vars)
            forecast_samples[i, :, :] = deterministic_forecast + sampled_residuals
            
        return forecast_samples
    
    def save_model(self, save_path: str) -> None:
        """
        Save VAR model and t-distribution parameters.
        
        Args:
            save_path: Path to save the model parameters
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save VAR model parameters
        var_params = {
            'coefs': self.var_results.coefs.tolist(),
            'intercept': self.var_results.intercept.tolist(),
            'k_ar': self.var_results.k_ar
        }
        
        # Save t-distribution parameters
        t_params_dict = {
            var: {'df': float(params['df']), 'loc': float(params['loc']), 'scale': float(params['scale'])}
            for var, params in self.t_params.items()
        }
        
        # Save all parameters
        params = {
            'var_params': var_params,
            't_params': t_params_dict
        }
        
        with open(save_path, 'w') as f:
            json.dump(params, f, indent=4)
    
    @classmethod
    def load_model(cls, load_path: str) -> 'VARUncertaintyEstimator':
        """
        Load saved VAR model and t-distribution parameters.
        
        Args:
            load_path: Path to load the model from
            
        Returns:
            VARUncertaintyEstimator: Loaded model instance
        """
        # Create new instance
        estimator = cls()
        
        # Load parameters
        with open(load_path, 'r') as f:
            params = json.load(f)
        
        # Create minimal VAR results object
        class MinimalVARResults:
            pass
        
        estimator.var_results = MinimalVARResults()
        estimator.var_results.coefs = np.array(params['var_params']['coefs'])
        estimator.var_results.intercept = np.array(params['var_params']['intercept'])
        estimator.var_results.k_ar = params['var_params']['k_ar']
        
        # Load t-distribution parameters
        estimator.t_params = params['t_params']
        
        return estimator

    def _generate_residual_samples(self, n_samples: int, n_vars: int) -> np.ndarray:
        """
        Generate residual samples from fitted Student-t distributions.
        
        Args:
            n_samples: Number of samples to generate
            n_vars: Number of variables
            
        Returns:
            np.ndarray: Generated residual samples
        """
        sampled_residuals = np.zeros((n_samples, n_vars))
        
        for j, var in enumerate(self.t_params):
            params = self.t_params[var]
            df = float(params['df'])
            loc = float(params['loc'])
            scale = float(params['scale'])
            
            # Generate samples from t-distribution
            t_samples = t.rvs(df, loc=loc, scale=scale, size=n_samples)
            sampled_residuals[:, j] = t_samples
            
        return sampled_residuals