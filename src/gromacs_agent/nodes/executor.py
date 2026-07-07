# src/gromacs_agent/nodes/executor.py
import os
from gromacs_agent.tools.gromacs_tools import GromacsTools
from gromacs_agent.core.state import AgentState

def execute_node(state: AgentState) -> dict:
    step = state["current_step"]
    tools = GromacsTools()
    
    # 簡易的なコマンドマッピング
    cmd_map = {
        "pdb2gmx": ["-f", "input.pdb", "-o", "processed.gro", "-water", "tip3p", "-ff", "amber99sb-ildn"],
        "em": ["grompp", "-f", "em.mdp", "-c", "processed.gro", "-p", "topol.top", "-o", "em.tpr"],
    }
    
    code, stdout, stderr = tools.run_gmx_command(step, cmd_map.get(step, []))
    
    return {
        # "RUNNING" から "SUCCESS" に修正
        "status": "SUCCESS" if code == 0 else "FAILED",
        "last_error": stderr if code != 0 else None,
        "log_snippet": stderr[-1000:] if code != 0 else None
    }
