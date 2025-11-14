from typing import Optional
from uuid import uuid4

class SessionManager:
    """Manage session state for fairness analysis"""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid4())
        self.train_file_path: Optional[str] = None
        self.test_file_path: Optional[str] = None
        self.model_file_path: Optional[str] = None
        self.target_column: Optional[str] = None
        self.analysis_results: Optional[dict] = None
        self.encoders: Optional[dict] = None
    
    def clear(self):
        """Clear all session data"""
        self.train_file_path = None
        self.test_file_path = None
        self.model_file_path = None
        self.target_column = None
        self.analysis_results = None
        self.encoders = None
    
    def is_ready_for_analysis(self) -> bool:
        """Check if session is ready for analysis"""
        return (
            self.train_file_path is not None and
            self.target_column is not None
        )

# Session storage (temporary solution - replace with Redis in production)
_sessions = {}

def get_session(session_id: str = None) -> SessionManager:
    """Dependency to get session"""
    if session_id is None:
        session_id = str(uuid4())
    
    if session_id not in _sessions:
        _sessions[session_id] = SessionManager(session_id)
    
    return _sessions[session_id]

# Keep global_session for backward compatibility (TEMPORARY)
global_session = get_session("default")