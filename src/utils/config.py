"""
Configuration management utilities for the Iris Classifier application.
"""

import yaml
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        raise 
