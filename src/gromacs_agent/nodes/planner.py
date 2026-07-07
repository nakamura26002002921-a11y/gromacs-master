# src/gromacs_agent/nodes/planner.py
from gromacs_agent.core.state import AgentState

def plan_node(state: AgentState) -> dict:
    """ユーザー要求から標準的なMDパイプラインを計画する"""
    workflow = ["pdb2gmx", "editconf", "solvate", "genion", "em", "nvt", "npt", "md"]
    return {
        "workflow": workflow,
        "current_step": workflow[0],
        "step_index": 0,
        "attempt_count": 0,
        "status": "PENDING"
    }
