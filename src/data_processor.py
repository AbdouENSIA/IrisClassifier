"""
Data processing utilities for the Iris Classifier application.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles all data processing operations."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.scaler = StandardScaler()

    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data format and values."""
        try:
            required_columns = ['sepal length (cm)', 'sepal width (cm)', 
                              'petal length (cm)', 'petal width (cm)']
            
            # Check for required columns
            if not all(col in data.columns for col in required_columns):
                return False
                
            # Check for numeric values
            if not data[required_columns].apply(pd.to_numeric, errors='coerce').notna().all().all():
                return False
                
            return True
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return False

    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess input data for prediction."""
        try:
            # Convert to numeric and handle missing values
            numeric_data = data.apply(pd.to_numeric, errors='coerce')
            numeric_data = numeric_data.fillna(numeric_data.mean())
            
            # Scale features
            scaled_data = self.scaler.fit_transform(numeric_data)
            
            return scaled_data
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise 