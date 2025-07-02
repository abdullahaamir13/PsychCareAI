import os
import joblib
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PsychiatricMLModels:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.model_metrics = {}
        self.feature_importance = None

    def load_models(self, directory='models/'):
        """Load models from saved files"""
        try:
            self.models.clear()
            self.scalers.clear()
            
            if not os.path.exists(directory):
                logger.error(f"Models directory not found: {directory}")
                return False
            
            # Load scaler
            scaler_path = os.path.join(directory, "scaler.pkl")
            if os.path.exists(scaler_path):
                self.scalers['main'] = joblib.load(scaler_path)
                logger.info(f"Loaded scaler from {scaler_path}")
            
            # Load label encoder
            le_path = os.path.join(directory, "label_encoder.pkl")
            if os.path.exists(le_path):
                self.label_encoders['main'] = joblib.load(le_path)
                logger.info(f"Loaded label encoder from {le_path}")
            
            # Load feature info
            feature_info_path = os.path.join(directory, "feature_info.pkl")
            if os.path.exists(feature_info_path):
                feature_info = joblib.load(feature_info_path)

                if isinstance(feature_info, dict):
                    self.feature_names = feature_info.get('feature_names', [])
                elif isinstance(feature_info, list):
                    self.feature_names = feature_info
                else:
                    logger.warning("Unexpected format in feature_info.pkl")
                        
            # Load model metrics
            metrics_path = os.path.join(directory, "model_evaluation.pkl")
            if os.path.exists(metrics_path):
                self.model_metrics = joblib.load(metrics_path)
                logger.info(f"Loaded model metrics")
            
            # Load models
            model_files = [f for f in os.listdir(directory) if f.endswith('_model.pkl')]
            for model_file in model_files:
                model_name = model_file.replace('_model.pkl', '').replace('_', ' ')
                model_path = os.path.join(directory, model_file)
                try:
                    model = joblib.load(model_path)
                    self.models[model_name] = model
                    logger.info(f"Loaded model {model_name} from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load {model_path}: {str(e)}")
                    continue
            
            if not self.models:
                logger.error("No models loaded successfully")
                return False
            
            logger.info(f"Successfully loaded {len(self.models)} models")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False

    def predict_disorder(self, patient_features):
        """Predict disorder for a patient using loaded models"""
        try:
            if not self.models:
                # Try to load models if not already loaded
                if not self.load_models():
                    return {'error': 'Models not available. Please train models first.'}

            if not patient_features or len(patient_features) == 0:
                return {'error': 'Input data must have at least one feature'}

            # Convert input to proper format
            if isinstance(patient_features, dict):
                # Convert dictionary to feature array based on expected features
                feature_array = self._dict_to_feature_array(patient_features)
            else:
                feature_array = np.array(patient_features).reshape(1, -1)

            # Scale features if scaler is available
            if 'main' in self.scalers:
                scaled_features = self.scalers['main'].transform(feature_array)
            else:
                scaled_features = feature_array

            predictions = {}
            for name, model in self.models.items():
                try:
                    probabilities = model.predict_proba(scaled_features)[0]
                    classes = model.classes_
                    
                    # Convert to original labels if label encoder is available
                    if 'main' in self.label_encoders:
                        original_classes = self.label_encoders['main'].inverse_transform(classes)
                    else:
                        original_classes = classes
                    
                    predictions[name] = dict(zip(original_classes, probabilities))
                except Exception as e:
                    logger.error(f"Error in prediction for {name}: {str(e)}")
                    continue

            return predictions

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {'error': f'Prediction failed: {str(e)}'}

    def _dict_to_feature_array(self, patient_dict):
        """Convert patient dictionary to feature array"""
        try:
            # Initialize feature array with zeros
            feature_array = np.zeros((1, len(self.feature_names)))
            
            # Map dictionary values to feature positions
            for i, feature_name in enumerate(self.feature_names):
                if feature_name in patient_dict:
                    value = patient_dict[feature_name]
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        feature_array[0, i] = value
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Error converting dict to feature array: {str(e)}")
            # Return zeros if conversion fails
            return np.zeros((1, len(self.feature_names)))

    def get_ensemble_prediction(self, patient_features):
        """Get ensemble prediction from all models"""
        try:
            individual_predictions = self.predict_disorder(patient_features)
            if 'error' in individual_predictions:
                return individual_predictions

            ensemble_probs = {}
            for model_name, preds in individual_predictions.items():
                for disorder, prob in preds.items():
                    ensemble_probs.setdefault(disorder, []).append(prob)

            final_predictions = {
                disorder: np.mean(probs) for disorder, probs in ensemble_probs.items()
            }

            return final_predictions

        except Exception as e:
            return {'error': f'Ensemble prediction failed: {str(e)}'}

    def get_risk_factors(self, patient_features):
        """Get risk factors based on feature importance"""
        try:
            if not self.models:
                return []

            # Get feature importance from first available model
            first_model = list(self.models.values())[0]
            if hasattr(first_model, 'feature_importances_'):
                feature_importance = first_model.feature_importances_
            else:
                return []

            if isinstance(patient_features, dict):
                feature_array = self._dict_to_feature_array(patient_features)[0]
            else:
                feature_array = np.array(patient_features)

            if len(feature_array) != len(self.feature_names):
                logger.warning(f"Feature length mismatch: expected {len(self.feature_names)}, got {len(feature_array)}")
                return []

            risk_factors = []
            for i, (feature, importance, value) in enumerate(zip(self.feature_names, feature_importance, feature_array)):
                if importance > 0.01 and abs(value) > 0.1:  # Lowered thresholds
                    risk_factors.append({
                        'feature': feature,
                        'importance': float(importance),
                        'value': float(value),
                        'risk_level': 'High' if abs(value) > 1.0 else 'Medium'
                    })

            return sorted(risk_factors, key=lambda x: x['importance'], reverse=True)[:10]

        except Exception as e:
            logger.error(f"Error getting risk factors: {str(e)}")
            return []

    def get_model_metrics(self):
        """Get model performance metrics"""
        try:
            # Try to load evaluation results
            evaluation_path = "models/model_evaluation.pkl"
            if os.path.exists(evaluation_path):
                return joblib.load(evaluation_path)
            else:
                return self.model_metrics
        except Exception as e:
            logger.error(f"Error loading model metrics: {str(e)}")
            return self.model_metrics
        
    def get_feature_importance(self, model_name=None):
        """Return feature importance scores for the specified model."""
        if not self.models:
            raise ValueError("No models loaded.")
        
        model_name = model_name or list(self.models.keys())[0]
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found.")
        
        # Try standard model types
        try:
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):
                return model.get_feature_importance()
            else:
                raise ValueError(f"Model '{model_name}' does not support feature importance.")
        except Exception as e:
            raise ValueError(f"Error retrieving feature importance for '{model_name}': {str(e)}")

    def get_available_models(self):
        """Get list of available models"""
        return list(self.models.keys())

    def get_feature_names(self):
        """Get list of feature names"""
        return self.feature_names.copy()

    def is_trained(self):
        """Check if models are trained and available"""
        return len(self.models) > 0 and 'main' in self.scalers
