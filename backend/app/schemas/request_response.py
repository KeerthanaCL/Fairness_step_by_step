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

class BiasAnalysisRequest(BaseModel):
    sensitive_feature_column: str = Field(..., description="Column name of sensitive attribute")
    prediction_column: Optional[str] = None
    prediction_proba_column: Optional[str] = None

class BiasMetricResult(BaseModel):
    metric: str
    is_biased: bool
    threshold: float
    description: str

class BiasAnalysisResponse(BaseModel):
    status: str
    analysis_type: str
    sensitive_feature: str
    total_metrics: int
    biased_metrics_count: int
    overall_bias_status: str
    metrics_results: Dict[str, Any]
    recommendations: List[str]

class PipelineRequest(BaseModel):
    """Request for mitigation pipeline - apply multiple strategies in sequence"""
    
    strategies: List[str] = Field(
        ...,
        description="List of mitigation strategies to apply in order",
        example=["disparate_impact_remover", "data_augmentation"]
    )
    
    sensitive_feature_column: str = Field(
        ...,
        description="Column name of the sensitive attribute to mitigate for (e.g., 'experience', 'gender', 'race')",
        example="experience"
    )
    
    prediction_column: Optional[str] = Field(
        None,
        description="Column name containing predictions (optional - if not provided, model will be used)",
        example=""
    )


class MitigationRequest(BaseModel):
    """Request for a single mitigation strategy"""
    
    strategy: str = Field(
        ...,
        description="Mitigation strategy to apply",
        example="disparate_impact_remover"
    )
    
    sensitive_feature_column: str = Field(
        ...,
        description="Column name of the sensitive attribute",
        example="experience"
    )
    
    prediction_column: Optional[str] = Field(
        None,
        description="Column name containing predictions",
        example=""
    )


class BiasDetectionResponse(BaseModel):
    """Response from bias detection"""
    
    status: str = Field(..., description="Status of detection (BIASED or FAIR)")
    biased_metrics_count: int = Field(..., description="Number of biased metrics found")
    biased_metrics: List[str] = Field(..., description="List of biased metrics")
    metrics: Dict[str, Any] = Field(..., description="Detailed metrics")


class MitigationResponse(BaseModel):
    """Response from mitigation"""
    
    status: str = Field(..., description="Success or error status")
    strategy_applied: str = Field(..., description="Strategy that was applied")
    bias_assessment: Dict[str, Any] = Field(..., description="Baseline and mitigated bias metrics")
    improvement_analysis: Dict[str, Any] = Field(..., description="Analysis of improvements")
    recommendations: Optional[List[str]] = Field(None, description="Recommendations for further improvement")


class PipelineResponse(BaseModel):
    """Response from mitigation pipeline"""
    
    status: str = Field(..., description="Overall pipeline status")
    strategies_applied: List[str] = Field(..., description="Strategies that were applied")
    initial_bias: Dict[str, Any] = Field(..., description="Initial bias metrics")
    final_bias: Dict[str, Any] = Field(..., description="Final bias metrics after all strategies")
    overall_improvement: int = Field(..., description="Number of biased metrics reduced")
    stage_results: Dict[str, Any] = Field(..., description="Results from each stage")
    recommendations: Optional[List[str]] = Field(None, description="Recommendations")