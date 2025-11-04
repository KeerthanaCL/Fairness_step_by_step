import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from app.config import UPLOADS_DIR

class FileManager:
    @staticmethod
    def save_uploaded_file(file_content: bytes, filename: str, file_type: str) -> str:
        """Save uploaded file to disk"""
        file_path = UPLOADS_DIR / f"{file_type}_{filename}"
        with open(file_path, "wb") as f:
            f.write(file_content)
        return str(file_path)
    
    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """Load CSV file"""
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise Exception(f"Error loading CSV: {str(e)}")
    
    @staticmethod
    def load_model(file_path: str):
        """Load pickle model"""
        try:
            with open(file_path, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    @staticmethod
    def get_existing_file(file_type: str) -> Optional[str]:
        """Get existing uploaded file path"""
        for file in UPLOADS_DIR.glob(f"{file_type}_*"):
            return str(file)
        return None
    
    @staticmethod
    def clear_files():
        """Clear all uploaded files"""
        for file in UPLOADS_DIR.glob("*"):
            if file.is_file():
                file.unlink()


class DataValidator:
    @staticmethod
    def validate_csv(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate CSV data"""
        if df.empty:
            return False, "CSV file is empty"
        if df.shape[1] == 0:
            return False, "CSV has no columns"
        return True, "CSV is valid"
    
    @staticmethod
    def validate_target_column(df: pd.DataFrame, target_column: str) -> Tuple[bool, str]:
        """Validate if target column exists"""
        if target_column not in df.columns:
            return False, f"Target column '{target_column}' not found in dataset"
        return True, "Target column is valid"
    
    @staticmethod
    def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Encode categorical features to numerical"""
        df_encoded = df.copy()
        encoding_map = {}
        
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                unique_values = df_encoded[col].unique()
                encoding_map[col] = {val: idx for idx, val in enumerate(unique_values)}
                df_encoded[col] = df_encoded[col].map(encoding_map[col])
        
        return df_encoded, encoding_map