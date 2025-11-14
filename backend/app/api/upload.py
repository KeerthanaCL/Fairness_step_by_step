from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from app.config import ALLOWED_CSV_EXTENSIONS, ALLOWED_MODEL_EXTENSIONS
from app.utils import FileManager, DataValidator
from app.models import global_session
from app.schemas.request_response import UploadResponse
import os

router = APIRouter(prefix="/api/upload", tags=["Upload"])

@router.post("/train", response_model=UploadResponse)
async def upload_train(file: UploadFile = File(...)):
    """Upload training CSV file"""
    try:
        # Validate file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_CSV_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Only CSV files are allowed for train data")
        
        # Read file content
        content = await file.read()
        
        # Save file
        file_path = FileManager.save_uploaded_file(content, file.filename, "train")
        global_session.train_file_path = file_path
        
        # Load and validate
        df = FileManager.load_csv(file_path)
        is_valid, message = DataValidator.validate_csv(df)
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        return UploadResponse(
            status="success",
            message="Training data uploaded successfully",
            train_shape=df.shape,
            columns=df.columns.tolist()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@router.post("/test", response_model=UploadResponse)
async def upload_test(file: UploadFile = File(...)):
    """Upload test CSV file"""
    try:
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_CSV_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Only CSV files are allowed for test data")
        
        content = await file.read()
        file_path = FileManager.save_uploaded_file(content, file.filename, "test")
        global_session.test_file_path = file_path
        
        df = FileManager.load_csv(file_path)
        is_valid, message = DataValidator.validate_csv(df)
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        return UploadResponse(
            status="success",
            message="Test data uploaded successfully",
            test_shape=df.shape,
            columns=df.columns.tolist()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@router.post("/model", response_model=UploadResponse)
async def upload_model(file: UploadFile = File(...)):
    """Upload trained model (PKL file)"""
    try:
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_MODEL_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Only PKL files are allowed for model")
        
        content = await file.read()
        file_path = FileManager.save_uploaded_file(content, file.filename, "model")
        global_session.model_file_path = file_path
        
        # Validate model can be loaded
        try:
            model = FileManager.load_model(file_path)
        except Exception as e:
            # Remove saved model file (avoid leaving bad file) and return clear error
            try:
                os.remove(file_path)
            except Exception:
                pass
            raise HTTPException(status_code=400, detail=f"Uploaded model could not be loaded: {str(e)}")
        
        return UploadResponse(
            status="success",
            message="Model uploaded successfully",
            columns=["model_loaded"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading model: {str(e)}")