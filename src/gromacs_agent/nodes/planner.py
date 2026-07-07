# src/gromacs_agent/nodes/planner.py
from gromacs_agent.core.state import AgentState

def plan_node(state: AgentState) -> dict:
    # すでにワークフローが設定されており、次のステップがある場合はそれを返す
    workflow = state.get("workflow", [])
    idx = state.get("step_index", 0)
    
    if workflow and idx < len(workflow):
        return {
            "current_step": workflow[idx],
            "attempt_count": 0,
            "status": "PENDING"
        }

    # 初回起動時のみワークフローを生成
    default_workflow = ["pdb2gmx", "editconf", "solvate", "genion", "em", "nvt", "npt", "md"]
    return {
        "workflow": default_workflow,
        "current_step": default_workflow[0],
        "step_index": 0,
        "attempt_count": 0,
        "status": "PENDING"
    }
