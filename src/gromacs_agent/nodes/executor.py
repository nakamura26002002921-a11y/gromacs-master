# src/gromacs_agent/nodes/executor.py
import os
from gromacs_agent.tools.gromacs_tools import GromacsTools
from gromacs_agent.core.state import AgentState

def execute_node(state: AgentState) -> dict:
    step = state["current_step"]
    tools = GromacsTools()
    
    # 簡易的なコマンドマッピング（実際は設定に応じて引数を組み立てる）
    cmd_map = {
        "pdb2gmx": ["-f", "input.pdb", "-o", "processed.gro", "-water", "tip3p", "-ff", "amber99sb-ildn"],
        "em": ["grompp", "-f", "em.mdp", "-c", "processed.gro", "-p", "topol.top", "-o", "em.tpr"],
        # ... 他も同様に
    }
    
    # 実際の実装では、state["current_config"]を参照してmdpファイルを生成/編集するロジックが入る
    code, stdout, stderr = tools.run_gmx_command(step, cmd_map.get(step, []))
    
    return {
        "status": "RUNNING" if code == 0 else "FAILED",
        "last_error": stderr if code != 0 else None,
        "log_snippet": stderr[-1000:] if code != 0 else None # 最後の1000文字を保持
    }
