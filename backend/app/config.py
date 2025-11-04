import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"

# Create uploads directory if it doesn't exist
UPLOADS_DIR.mkdir(exist_ok=True)

# Configuration
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_CSV_EXTENSIONS = {".csv"}
ALLOWED_MODEL_EXTENSIONS = {".pkl"}

# HSIC Configuration
HSIC_KERNEL = "rbf"
HSIC_THRESHOLD_PERCENTILE = 50  # Use median as threshold