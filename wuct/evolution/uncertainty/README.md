# Uncertainty Estimation

This package provides uncertainty estimation methods for evolution prediction.

## Available Estimators

### Vector Autoregression (VAR)

The VAR-based uncertainty estimator combines Vector Autoregression with Student's t-distribution
to model forecast uncertainties.

#### Basic Usage

python
```
from moct.evolution.uncertainty import UncertaintyEstimatorFactory

# Create estimator
estimator = UncertaintyEstimatorFactory.create("var", n_samples=1000)

# Train the model
# training_data shape: (n_timesteps, n_variables)
estimator.fit(training_data, train_size=0.8)

# Generate samples for new data
# test_data shape: (n_steps, n_variables)
samples = estimator.generate_samples(test_data)

# samples shape: (n_steps, n_samples, n_variables)

# Save trained model
estimator.save_model("models/var_model.json")

# Load saved model
loaded_estimator = VARUncertaintyEstimator.load_model("models/var_model.json")
```

#### Advanced Usage

[Add more detailed examples, parameter tuning, etc.]

### Random Forest (RF)

[Similar documentation for RF estimator]

## Adding New Estimators

To add a new uncertainty estimator:

1. Create a new class that inherits from `UncertaintyEstimator`
2. Implement the required methods: `fit()`, `generate_samples()`, `save_model()`, `load_model()`
3. Register the estimator using the `@UncertaintyEstimatorFactory.register()` decorator

Example:
python
```
@UncertaintyEstimatorFactory.register("my_estimator")
class MyEstimator(UncertaintyEstimator):
def init(self, n_samples: int = 1000):
super().init(n_samples)
# Additional initialization
```

### Random Forest (RF)

Example usage:
```
# Prepare data (features and target in last column)
data = np.column_stack([features, target])

# Create and train estimator
estimator = UncertaintyEstimatorFactory.create(
    "rf", 
    n_samples=1000, 
    n_estimators=100
)
estimator.fit(data)

# Generate samples for new data
samples = estimator.generate_samples(new_data)

# Save and load
estimator.save_model("models/rf_model.joblib")
loaded_estimator = RFUncertaintyEstimator.load_model("models/rf_model.joblib")
```