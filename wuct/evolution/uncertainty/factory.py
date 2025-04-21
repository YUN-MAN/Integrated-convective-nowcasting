from typing import Type, Dict
from .base import UncertaintyEstimator

class UncertaintyEstimatorFactory:
    _estimators: Dict[str, Type[UncertaintyEstimator]] = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(estimator_cls: Type[UncertaintyEstimator]):
            cls._estimators[name] = estimator_cls
            return estimator_cls
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs) -> UncertaintyEstimator:
        if name not in cls._estimators:
            raise ValueError(f"Unknown estimator: {name}")
        return cls._estimators[name](**kwargs)