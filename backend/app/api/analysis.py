from fastapi import APIRouter, HTTPException
from app.utils import FileManager
from app.hsic_detector import AnalysisEngine
from app.models import global_session
from app.schemas.request_response import AnalysisResponse, FeatureSensitivityScore
import logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analysis", tags=["Analysis"])

@router.post("/run", response_model=AnalysisResponse)
async def run_fairness_analysis(n_top_features: int = 10):
    """
    Run fairness analysis using HSIC Lasso (pyHSICLasso library)
    
    Args:
        n_top_features: Number of top features to select (default: 10)
    """
    try:
        # Validate preconditions
        if not global_session.is_ready_for_analysis():
            raise HTTPException(
                status_code=400, 
                detail="Training data and target column must be configured first"
            )
        
        # Load data
        train_df = FileManager.load_csv(global_session.train_file_path)
        
        # Initialize analysis engine
        engine = AnalysisEngine()
        engine.train_df = train_df
        engine.target_column = global_session.target_column
        
        # Run analysis
        results = engine.run_analysis(n_top_features=n_top_features)
        results["detector_obj"] = engine.detector
        
        # Store results in session
        global_session.analysis_results = results
        
        # Convert to response format
        feature_sensitivity_scores = [
            FeatureSensitivityScore(
                feature=score["feature"],
                nocco_score=score["nocco_score"],
                is_sensitive=score["is_sensitive"],
                percentile_rank=score["percentile_rank"]
            )
            for score in results["feature_sensitivity_scores"]
        ]

        logger.info("Analysis finished; stored detector instance in session")
        
        return AnalysisResponse(
            status="success",
            analysis_type=results["analysis_summary"]["analysis_type"],
            target_column=global_session.target_column,
            total_features=results["analysis_summary"]["total_features"],
            sensitive_features_count=results["analysis_summary"]["sensitive_features_count"],
            threshold_value=results["threshold_value"],
            feature_scores=feature_sensitivity_scores,
            sensitive_features=results["sensitive_features"],
            analysis_summary=results["analysis_summary"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running analysis: {str(e)}")


@router.get("/results")
async def get_analysis_results():
    """Get stored analysis results"""
    try:
        if global_session.analysis_results is None:
            raise HTTPException(
                status_code=404, 
                detail="No analysis results found. Run analysis first."
            )
        
        # Create a copy and exclude non-serializable objects
        results_copy = {
            key: value 
            for key, value in global_session.analysis_results.items() 
            if key != "detector_obj"
        }
        
        return {
            "status": "success",
            "results": results_copy
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving results: {str(e)}")


@router.get("/feature-relationships/{feature_index}")
async def get_feature_relationships(feature_index: int, num_neighbors: int = 5):
    """
    Get features that are related to a specific feature using the already-fitted detector.
    This avoids re-running the HSIC analysis (which is expensive and may give different results).
    
    Args:
        feature_index: Index of the feature (0-based)
        num_neighbors: Number of related features to return
    """
    try:
        if global_session.analysis_results is None:
            raise HTTPException(
                status_code=404, 
                detail="No analysis results found. Run analysis first."
            )
        
        # Retrieve detector instance stored during initial analysis
        detector = global_session.analysis_results.get("detector_obj", None)
        if detector is None:
            raise HTTPException(
                status_code=500,
                detail="Detector instance not found in session. Re-run analysis to rebuild detector."
            )

        # Validate feature index range
        total_features = global_session.analysis_results.get("analysis_summary", {}).get("total_features", None)
        if total_features is not None and (feature_index < 0 or feature_index >= total_features):
            raise HTTPException(status_code=400, detail=f"feature_index {feature_index} out of range (0..{total_features-1})")

        relationships = detector.get_feature_relationships(feature_index, num_neighbors)

        return {
            "status": "success",
            "relationships": relationships
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error retrieving feature relationships")
        raise HTTPException(status_code=500, detail=f"Error retrieving relationships: {str(e)}")