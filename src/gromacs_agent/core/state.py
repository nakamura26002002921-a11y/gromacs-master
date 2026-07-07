# src/gromacs_agent/core/state.py
from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator

class AgentState(TypedDict):
    """GROMACS Agentの状態遷移モデル"""
    # システム情報
    system_name: str
    pdb_file: str
    
    # ワークフロー
    workflow: List[str]  # ["em", "nvt", "npt", "md"]
    current_step: str
    step_index: int
    
    # 実行状態
    attempt_count: int
    max_attempts: int
    status: str  # "PENDING", "RUNNING", "SUCCESS", "FAILED", "NEEDS_REPLAN"
    
    # データ
    last_error: Optional[str]
    log_snippet: Optional[str]
    current_config: Dict[str, Any]
    
    # 蓄積されたナレッジ（診断用）
    diagnosis_context: Optional[str]
    
    # 履歴（Reflection用）
    history: List[Dict[str, Any]]
