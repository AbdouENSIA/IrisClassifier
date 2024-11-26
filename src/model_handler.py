"""
Model handling utilities for the Iris Classifier application.
"""

import pickle
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from sklearn.base import BaseEstimator
import logging

logger = logging.getLogger(__name__)

class ModelHandler:
    """Handles all model-related operations."""
    
    def __init__(self, model_path: Path):
        """Initialize the model handler."""
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self) -> BaseEstimator:
        """Load the trained model from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions using the loaded model."""
        try:
            return self.model.predict(features)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        try:
            return self.model.predict_proba(features)
        except Exception as e:
            logger.error(f"Probability prediction failed: {str(e)}")
            raise 