# src/gromacs_agent/nodes/planner.py
import tempfile

from gromacs_agent.core.state import AgentState
from gromacs_agent.utils.command_logger import reset_script


def plan_node(state: AgentState) -> dict:
    # すでにワークフローが設定されており、次のステップがある場合はそれを返す
    workflow = state.get("workflow", [])
    idx = state.get("step_index", 0)

    if workflow and idx < len(workflow):
        return {
            "current_step": workflow[idx],
            "attempt_count": 0,
            "status": "PENDING",
        }

    # 初回起動時のみワークフローとwork_dirを生成
    default_workflow = ["pdb2gmx", "editconf", "solvate", "genion", "em", "nvt", "npt", "md"]

    system_name = state.get("system_name", "run")
    work_dir = state.get("work_dir") or tempfile.mkdtemp(prefix=f"gmx_agent_{system_name}_")
    # 新規runの開始時点でreproduce.shを空にしておく (過去実行分の混入を防ぐ)
    reset_script(work_dir)

    return {
        "workflow": default_workflow,
        "current_step": default_workflow[0],
        "step_index": 0,
        "attempt_count": 0,
        "status": "PENDING",
        "work_dir": work_dir,
    }
