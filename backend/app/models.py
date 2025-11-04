from typing import Optional

class SessionManager:
    """Manage session state for fairness analysis"""
    
    def __init__(self):
        self.train_file_path: Optional[str] = None
        self.test_file_path: Optional[str] = None
        self.model_file_path: Optional[str] = None
        self.target_column: Optional[str] = None
        self.analysis_results: Optional[dict] = None
    
    def clear(self):
        """Clear all session data"""
        self.train_file_path = None
        self.test_file_path = None
        self.model_file_path = None
        self.target_column = None
        self.analysis_results = None
    
    def is_ready_for_analysis(self) -> bool:
        """Check if session is ready for analysis"""
        return (
            self.train_file_path is not None and
            self.target_column is not None
        )

# Global session
global_session = SessionManager()