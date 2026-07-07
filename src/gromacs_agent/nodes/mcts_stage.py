# src/gromacs_agent/nodes/mcts_stage.py
from gromacs_agent.core.state import AgentState
from gromacs_agent.mcts.search import MCTSStageSearch
from gromacs_agent.knowledge.db import KnowledgeBase
from gromacs_agent.nodes.executor import execute_node


def mcts_stage_node(state: AgentState) -> dict:
    """
    MCTSを使って現在のステージ (em, nvt等) の最適パラメータを探索するノード。
    """
    stage = state["current_step"]
    base_config = state["current_config"]
    kb = KnowledgeBase()

    def simulate_fn(stage_name: str, config: dict) -> tuple[bool, float, str]:
        temp_state = {**state, "current_config": config}
        result = execute_node(temp_state)
        success = result["status"] == "SUCCESS"
        reward = 1.0 if success else 0.0
        log = result.get("last_error", "") or ""
        return success, reward, log

    search = MCTSStageSearch(
        stage=stage,
        base_config=base_config,
        kb=kb,
        simulate_fn=simulate_fn,
        max_iterations=state.get("mcts_max_iterations", 4),
    )
    result = search.run()

    success = result["success"]

    # テストが期待するフォーマットで history エントリを作成
    mcts_entry = {
        "type": "mcts",
        "stage": stage,
        "success": success,
        "iterations_used": result["iterations_used"],
        "config": result["config"],
        "tree": result.get("tree", {}),
    }

    return {
        # ★ テストは "STAGE_FAILED" を期待している
        "status": "SUCCESS" if success else "STAGE_FAILED",
        "current_config": result["config"],
        "last_error": result["log"] if not success else None,
        "history": state.get("history", []) + [mcts_entry],
    }
