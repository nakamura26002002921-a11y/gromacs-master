# src/gromacs_agent/core/state.py
from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    """LangGraphで管理するエージェントの状態"""
    system_name: str
    pdb_file: str
    workflow: List[str]
    current_step: str
    step_index: int
    attempt_count: int
    max_attempts: int
    status: str  # "PENDING", "RUNNING", "SUCCESS", "FAILED", "STAGE_FAILED", "ALL_DONE"
    last_error: Optional[str]
    log_snippet: Optional[str]
    current_config: Dict[str, Any]
    diagnosis_context: Optional[Any]
    history: List[Dict[str, Any]]
    mcts_stages: List[str]           # ← 追加
    mcts_max_iterations: int         # ← 追加
