# src/gromacs_agent/nodes/replanner.py
from gromacs_agent.core.state import AgentState

def replan_node(state: AgentState) -> dict:
    diagnosis = state["diagnosis_context"]
    new_config = state["current_config"].copy()
    
    if diagnosis["fix_type"] == "PARAMETER_CHANGE":
        # 例: dtを小さくする
        new_config["dt"] = diagnosis["parameters"].get("dt", 0.001)
    
    return {
        "current_config": new_config,
        "attempt_count": state["attempt_count"] + 1,
        "status": "PENDING" # 再実行キュー
    }
