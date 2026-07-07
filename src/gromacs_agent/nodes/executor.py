# src/gromacs_agent/nodes/executor.py
import os
from gromacs_agent.tools.gromacs_tools import GromacsTools
from gromacs_agent.core.state import AgentState

def execute_node(state: AgentState) -> dict:
    step = state["current_step"]
    config = state["current_config"]
    tools = GromacsTools()

    # 1. current_config から .mdp ファイルを動的生成
    mdp_lines = []
    for k, v in config.items():
        if not isinstance(v, dict): # ネストされた設定は除外
            mdp_lines.append(f"{k} = {v}")
    
    mdp_file = f"{step}.mdp"
    with open(mdp_file, "w") as f:
        f.write("\n".join(mdp_lines))

    # 2. grompp 実行
    grompp_args = ["-f", mdp_file, "-c", "processed.gro", "-p", "topol.top", "-o", f"{step}.tpr"]
    code, out, err = tools.run_gmx_command("grompp", grompp_args)
    if code != 0:
        return {"status": "FAILED", "last_error": err, "log_snippet": err[-1000:]}

    # 3. mdrun 実行
    mdrun_args = ["-deffnm", step, "-s", f"{step}.tpr"]
    code, out, err = tools.run_gmx_command("mdrun", mdrun_args)

    return {
        "status": "SUCCESS" if code == 0 else "FAILED",
        "last_error": err if code != 0 else None,
        "log_snippet": err[-1000:] if code != 0 else None
    }
