# app.py
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Custom modules
from src.model_handler import ModelHandler
from src.data_processor import DataProcessor
from src.utils.config import load_config
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger()

class IrisClassifierApp:
    """Main application class for the Iris Classifier web application."""

    def __init__(self):
        """Initialize the application with configurations and models."""
        try:
            # Load configurations
            self.config = load_config()
            
            # Setup paths
            self.model_path = Path(self.config['model']['path'])
            self.assets_path = Path(self.config['app']['assets_path'])
            
            # Initialize components
            self.model_handler = ModelHandler(self.model_path)
            self.data_processor = DataProcessor()
            
            # Load dataset information
            self.iris = load_iris()
            self.feature_names = self.iris.feature_names
            self.target_names = self.iris.target_names
            
            # Initialize session state
            self._initialize_session_state()
            
        except Exception as e:
            logger.error(f"Failed to initialize app: {str(e)}")
            st.error("Application failed to initialize. Please contact support.")
            sys.exit(1)

    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        if 'total_predictions' not in st.session_state:
            st.session_state.total_predictions = 0

    def setup_page_config(self) -> None:
        """Configure the Streamlit page settings."""
        st.set_page_config(
            page_title="Professional Iris Classifier",
            page_icon="🌸",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def render_header(self) -> None:
        """Render the application header with title and description."""
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("🌸 Professional Iris Classification System")
            st.markdown("""
                This enterprise-grade application provides accurate classification of Iris flowers
                using machine learning. Upload your own data or use the interactive feature input.
            """)
        with col2:
            if (self.assets_path / "logo.png").exists():
                st.image(str(self.assets_path / "logo.png"), width=150)

    def render_sidebar(self) -> Dict[str, float]:
        """Render and handle sidebar inputs."""
        st.sidebar.header("Feature Input")
        
        input_method = st.sidebar.radio(
            "Select Input Method",
            ["Interactive Sliders", "Manual Input", "File Upload"]
        )
        
        if input_method == "Interactive Sliders":
            return self._handle_slider_input()
        elif input_method == "Manual Input":
            return self._handle_manual_input()
        else:
            return self._handle_file_upload()

    def _handle_slider_input(self) -> Dict[str, float]:
        """Handle slider-based input method."""
        features = {}
        with st.sidebar.form("feature_form"):
            for feature in self.feature_names:
                features[feature] = st.slider(
                    feature,
                    float(self.config['features']['min_values'][feature]),
                    float(self.config['features']['max_values'][feature]),
                    float(self.config['features']['default_values'][feature]),
                    help=self.config['features']['descriptions'][feature]
                )
            submit_button = st.form_submit_button("Classify")
        
        return features if submit_button else None

    def visualize_results(self, features: Dict[str, float], prediction: int, probabilities: np.ndarray) -> None:
        """Create and display visualizations for the classification results."""
        col1, col2 = st.columns(2)
        
        with col1:
            # Probability bar chart
            fig_prob = go.Figure(data=[
                go.Bar(
                    x=self.target_names,
                    y=probabilities[0],
                    marker_color=['#FF9999', '#66B2FF', '#99FF99']
                )
            ])
            fig_prob.update_layout(
                title="Classification Probabilities",
                xaxis_title="Species",
                yaxis_title="Probability",
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig_prob)

        with col2:
            # Feature comparison
            feature_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Value': list(features.values())
            })
            fig_features = px.bar(
                feature_df,
                x='Feature',
                y='Value',
                title="Feature Values Comparison"
            )
            st.plotly_chart(fig_features)

    def save_prediction(self, features: Dict[str, float], prediction: str, confidence: float) -> None:
        """Save prediction to session state and optionally to database."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prediction_record = {
            'timestamp': timestamp,
            'features': features,
            'prediction': prediction,
            'confidence': confidence
        }
        st.session_state.prediction_history.append(prediction_record)
        st.session_state.total_predictions += 1

    def run(self) -> None:
        """Run the main application loop."""
        try:
            self.setup_page_config()
            self.render_header()

            # Sidebar inputs
            features = self.render_sidebar()

            if features:
                # Process input and make prediction
                input_array = np.array(list(features.values())).reshape(1, -1)
                prediction = self.model_handler.predict(input_array)
                probabilities = self.model_handler.predict_proba(input_array)

                # Display results
                st.write("### Classification Results")
                predicted_class = self.target_names[prediction[0]]
                confidence = max(probabilities[0]) * 100

                st.success(f"Predicted Class: **{predicted_class}** (Confidence: {confidence:.2f}%)")
                
                # Visualizations
                self.visualize_results(features, prediction[0], probabilities)
                
                # Save prediction
                self.save_prediction(features, predicted_class, confidence)

                # Display prediction history
                if st.checkbox("Show Prediction History"):
                    self.display_prediction_history()

        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            st.error("An error occurred while processing your request. Please try again.")

    def display_prediction_history(self) -> None:
        """Display the history of predictions made in the current session."""
        if st.session_state.prediction_history:
            st.write("### Prediction History")
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df)
            
            # Download button for prediction history
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download Prediction History",
                data=csv,
                file_name="prediction_history.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    app = IrisClassifierApp()
    app.run()
