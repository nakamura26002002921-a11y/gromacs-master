# src/gromacs_agent/core/state.py
from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """LangGraphで管理するエージェントの状態"""
    system_name: str
    pdb_file: str
    workflow: List[str]
    current_step: str
    step_index: int
    attempt_count: int
    max_attempts: int
    status: str  # "PENDING", "RUNNING", "SUCCESS", "FAILED"
    last_error: Optional[str]
    log_snippet: Optional[str]
    current_config: Dict[str, Any]
    diagnosis_context: Optional[str]
    history: List[Dict[str, Any]]
