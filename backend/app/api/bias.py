from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Optional, List, Dict
import pickle
import numpy as np
import pandas as pd
from app.utils import FileManager
from app.models import global_session
from app.bias_detector import BiasDetector
from app.schemas.request_response import BiasAnalysisRequest, BiasAnalysisResponse
import logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/bias", tags=["Bias Detection"])

@router.post("/analyze")
async def analyze_bias(request: BiasAnalysisRequest):
    """
    Analyze bias using fairness metrics
    """
    try:
        if global_session.train_file_path is None:
            raise HTTPException(status_code=400, detail="Training data must be uploaded first")
        
        # Load data
        train_df = FileManager.load_csv(global_session.train_file_path)
        
        # Get y_true
        if global_session.target_column is None:
            raise HTTPException(status_code=400, detail="Target column must be set first")
        
        y_true = train_df[global_session.target_column].values
        y_true = _safe_convert_to_numeric(y_true, "y_true")
        
        # Get sensitive feature
        if request.sensitive_feature_column not in train_df.columns:
            raise HTTPException(status_code=400, detail=f"Sensitive feature '{request.sensitive_feature_column}' not found")
        
        sensitive_feature = train_df[request.sensitive_feature_column].values
        print(f"\n[DEBUG] Original sensitive_feature type: {type(sensitive_feature[0])}, sample values: {sensitive_feature[:5]}")
        sensitive_feature = _safe_convert_to_numeric(sensitive_feature, request.sensitive_feature_column)
        print(f"[DEBUG] Converted sensitive_feature: {sensitive_feature[:5]}")
        
        # Get predictions
        if request.prediction_column:
            if request.prediction_column not in train_df.columns:
                raise HTTPException(status_code=400, detail=f"Prediction column '{request.prediction_column}' not found")
            y_pred = train_df[request.prediction_column].values
            y_pred = _safe_convert_to_numeric(y_pred, request.prediction_column)

        elif global_session.model_file_path:
            # Use uploaded model to predict
            model = FileManager.load_model(global_session.model_file_path)
            X = train_df.drop(columns=[global_session.target_column]).copy()
    
            # Encode categorical columns in X (same way model expects)
            print(f"\n[DEBUG] Encoding X data for model prediction")
            print(f"  X dtypes before: {X.dtypes.unique()}")
            
            X = _encode_dataframe(X)
            
            print(f"  X dtypes after: {X.dtypes.unique()}")
            print(f"  X shape: {X.shape}")
            
            y_pred = model.predict(X)
            y_pred = _safe_convert_to_numeric(y_pred, "model_prediction")
        else:
            raise HTTPException(status_code=400, detail="Either prediction_column or uploaded model required")
        
        # Get prediction probabilities if available
        y_pred_proba = None
        if request.prediction_proba_column:
            if request.prediction_proba_column in train_df.columns:
                try:
                    y_pred_proba = train_df[request.prediction_proba_column].values.astype(float)
                except:
                    y_pred_proba = None

        print(f"\n[INFO] Bias Analysis Input:")
        print(f"  y_true shape: {y_true.shape}, unique values: {np.unique(y_true)}")
        print(f"  y_pred shape: {y_pred.shape}, unique values: {np.unique(y_pred)}")
        print(f"  sensitive_feature shape: {sensitive_feature.shape}, unique values: {np.unique(sensitive_feature)}")
        
        # Run bias detection
        detector = BiasDetector()
        results = detector.run_all_metrics(y_true, y_pred, sensitive_feature, y_pred_proba)

        # Clean NaN values from results
        results = _clean_nan_values(results)
        
        # Generate recommendations
        recommendations = _generate_recommendations(results)
        
        return {
            'status': 'success',
            'analysis_type': 'Fairness Bias Detection',
            'sensitive_feature': request.sensitive_feature_column,
            'total_metrics': results['summary']['total_metrics'],
            'biased_metrics_count': results['summary']['biased_metrics_count'],
            'overall_bias_status': results['summary']['overall_bias_status'],
            'metrics_results': {k: v for k, v in results.items() if k != 'summary'},
            'recommendations': recommendations
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Error in analyze_bias: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error analyzing bias: {str(e)}")

@router.get("/thresholds")
async def get_bias_thresholds():
    """Get current bias detection thresholds"""
    detector = BiasDetector()
    return {
        'status': 'success',
        'thresholds': detector.thresholds,
        'descriptions': {
            'statistical_parity': 'Max acceptable difference in positive prediction rate',
            'disparate_impact': 'Min acceptable ratio (80% rule)',
            'equal_opportunity': 'Max acceptable difference in TPR',
            'equalized_odds': 'Max acceptable difference in TPR and FPR',
            'calibration': 'Max acceptable calibration error',
            'generalized_entropy_index': 'Max acceptable entropy index'
        }
    }

def _safe_convert_to_numeric(data: np.ndarray, name: str) -> np.ndarray:
    """
    Safely convert any data to numeric integers.
    - Preserves order-based mapping for categorical values (first-seen -> 0..n-1).
    - Handles booleans, ints, floats, strings and mixed object arrays.
    - Raises a ValueError if conversion fails.
    """
    logger.info(f"[CONVERT] Converting {name}")
    
    try:
        # Already numeric floats/ints -> cast to int safely
        if np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.floating):
            return data.astype(int)

        # Boolean -> int
        if data.dtype == bool or (data.dtype == np.bool_):
            return data.astype(int)
        
        # Object - possibly strings / mixed
        if data.dtype == object or data.dtype == 'O':
            
            # Try direct numeric conversion first
            try:
                numeric = pd.to_numeric(data, errors='raise')
                return numeric.astype(int).to_numpy()
            except Exception:
                # Fall back to stable categorical mapping (preserve first-seen order)
                mapping = {}
                mapped = []
                next_idx = 0
                for val in data:
                    key = val if not (isinstance(val, str) and val.strip() == "") else "__MISSING__"
                    if key not in mapping:
                        mapping[key] = next_idx
                        next_idx += 1
                    mapped.append(mapping[key])
                logger.debug(f"[CONVERT] {name} categorical mapping: {mapping}")
                return np.array(mapped, dtype=int)

        # As a last resort, attempt converting element-wise
        return np.array([int(x) for x in data], dtype=int)

    except Exception as e:
        logger.exception(f"Conversion failed for {name}")
        raise ValueError(f"Could not convert {name} to numeric: {str(e)}")
    
def _encode_dataframe(df: pd.DataFrame, saved_encoders: Dict[str, Dict] = None) -> pd.DataFrame:
    """
    Encode categorical columns to numeric in a stable, reproducible way.
    - If `saved_encoders` provided (dict of col -> mapping), reuse them.
      Otherwise, create mappings where needed (preserving first-seen order).
    - Returns df_encoded (numeric) and *does not* alter original df.
    """
    df_encoded = df.copy()
    encoders = saved_encoders or {}
    
    for col in df_encoded.columns:
        # If dtype numeric already, ensure numeric and fill NAs
        if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.name == 'category':
            df_encoded[col] = df_encoded[col].astype(str)

            if col in encoders:
                # Use saved encoder
                le = encoders[col]
                # Handle unseen labels
                df_encoded[col] = df_encoded[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            else:
                # Create new encoder
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                encoders[col] = le
        
        # CRITICAL: Ensure all columns are 1D numeric arrays
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(0)
    
    # VALIDATE: Check for homogeneous shape
    try:
        test_array = df_encoded.values
        if test_array.dtype == 'object':
            # Convert object arrays to float
            df_encoded = df_encoded.astype(float)
    except Exception as e:
        logger.error(f"Error converting to numpy array: {e}")
        # Force conversion column by column
        for col in df_encoded.columns:
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(0)
    
    return df_encoded, encoders

def _clean_nan_values(obj):
    """Recursively remove NaN values from nested dicts/lists"""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if isinstance(v, float) and np.isnan(v):
                cleaned[k] = 0.0
            elif isinstance(v, (dict, list)):
                cleaned[k] = _clean_nan_values(v)
            else:
                cleaned[k] = v
        return cleaned
    elif isinstance(obj, list):
        return [_clean_nan_values(item) for item in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return 0.0
    return obj

def _generate_recommendations(results: Dict) -> List[str]:
    """Generate recommendations based on bias analysis"""
    recommendations = []
    
    if results['summary']['overall_bias_status'] == 'FAIR':
        recommendations.append('✅ Model appears fair across selected metrics')
        return recommendations
    
    biased = results['summary']['biased_metrics']
    
    metrics_results = {k: v for k, v in results.items() if k != 'summary'}

    if 'statistical_parity' in biased:
        sp = metrics_results.get('statistical_parity', {})
        recommendations.append(f"⚠️ Statistical Parity violated (diff: {sp.get('difference', 0):.3f}) - Different prediction rates between groups. Consider resampling or reweighting data.")
    
    if 'disparate_impact' in biased:
        di = metrics_results.get('disparate_impact', {})
        recommendations.append(f"⚠️ Disparate Impact detected (ratio: {di.get('ratio', 0):.3f}) - Selection rate ratio < 0.80. Review decision thresholds.")
    
    if 'equal_opportunity' in biased:
        eo = metrics_results.get('equal_opportunity', {})
        if eo.get('status') != 'skipped':
            recommendations.append(f"⚠️ Equal Opportunity violated - True Positive Rates differ. Adjust decision thresholds.")
    
    if 'equalized_odds' in biased:
        eq = metrics_results.get('equalized_odds', {})
        if eq.get('status') != 'skipped':
            recommendations.append(f"⚠️ Equalized Odds violated - Apply fairness-aware post-processing.")
    
    if 'calibration' in biased:
        cal = metrics_results.get('calibration', {})
        recommendations.append(f"⚠️ Calibration error - Predicted probabilities differ from actual rates. Recalibrate model.")
    
    if 'generalized_entropy_index' in biased:
        recommendations.append(f"⚠️ High inequality in predictions - Apply fairness constraints during training.")
    
    return recommendations