from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class UploadResponse(BaseModel):
    status: str
    message: str
    train_shape: Optional[tuple] = None
    test_shape: Optional[tuple] = None
    columns: Optional[List[str]] = None

class TargetColumnRequest(BaseModel):
    target_column: str = Field(..., description="Name of the target column")

class ConfigurationResponse(BaseModel):
    status: str
    message: str
    configured: bool

class AnalysisRequest(BaseModel):
    n_top_features: int = Field(10, description="Number of top features to select")

class FeatureSensitivityScore(BaseModel):
    feature: str
    nocco_score: float
    is_sensitive: bool
    percentile_rank: float
    test_type: str = "unknown"

class AnalysisResponse(BaseModel):
    status: str
    analysis_type: str
    target_column: str
    total_features: int
    sensitive_features_count: int
    threshold_value: float
    feature_scores: List[FeatureSensitivityScore]
    sensitive_features: List[str]
    analysis_summary: Dict[str, Any]

class StatusResponse(BaseModel):
    status: str
    train_uploaded: bool
    test_uploaded: bool
    model_uploaded: bool
    target_column: Optional[str] = None
    train_shape: Optional[tuple] = None
    test_shape: Optional[tuple] = None
    columns: Optional[List[str]] = None

class FeatureRelationshipResponse(BaseModel):
    primary_feature: str
    related_features: List[Dict[str, Any]]