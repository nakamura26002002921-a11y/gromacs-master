# src/gromacs_agent/core/graph.py
from langgraph.graph import StateGraph, END
from gromacs_agent.core.state import AgentState
from gromacs_agent.nodes.planner import plan_node
from gromacs_agent.nodes.executor import execute_node
from gromacs_agent.nodes.diagnoser import diagnose_node
from gromacs_agent.nodes.replanner import replan_node
from gromacs_agent.nodes.mcts_stage import mcts_stage_node


def _route_after_planner(state: AgentState) -> str:
    current = state.get("current_step", "")
    mcts_stages = state.get("mcts_stages", [])
    if current in mcts_stages:
        return "mcts"
    return "execute"


def _route_after_executor(state: AgentState) -> str:
    if state.get("status") == "SUCCESS":
        return "advance"
    return "diagnose"


def _route_after_mcts(state: AgentState) -> str:
    if state.get("status") == "SUCCESS":
        return "advance"
    return "end_fail"


def _route_after_replanner(state: AgentState) -> str:
    if state.get("attempt_count", 0) < state.get("max_attempts", 3):
        return "retry"
    return "end_fail"


def advance_stage_node(state: AgentState) -> dict:
    workflow = state.get("workflow", [])
    next_index = state.get("step_index", 0) + 1

    if next_index >= len(workflow):
        return {"status": "ALL_DONE", "step_index": next_index}

    return {
        "step_index": next_index,
        "current_step": workflow[next_index],
        "status": "PENDING",
        "attempt_count": 0,
        "last_error": None,
    }


def stage_failed_node(state: AgentState) -> dict:
    """
    replanner/mctsが予算(max_attempts)を使い切って回復できなかった場合の終端処理。
    replan_nodeは「再実行キュー」を表すために常にstatus="PENDING"を返すため、
    それをそのままENDに渡すとFinal Statusが誤って"PENDING"になってしまう。
    ここで明示的にSTAGE_FAILEDへ上書きしてから終了する。
    """
    return {"status": "STAGE_FAILED"}


def build_graph():
    workflow = StateGraph(AgentState)

    # 1. ノードの追加
    workflow.add_node("planner", plan_node)
    workflow.add_node("executor", execute_node)
    workflow.add_node("mcts_stage", mcts_stage_node)
    workflow.add_node("diagnoser", diagnose_node)
    workflow.add_node("replanner", replan_node)

    # 2. エントリーポイントと最初のエッジ
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")

    # 3. executor の後のルーティングロジック
    def route_after_executor(state):
        # 失敗したら Diagnoser (LangChain) へ
        if state.get("status") == "FAILED":
            return "diagnoser"
        
        # MCTS対象ステージ (em, nvtなど) なら MCTS探索へ
        mcts_stages = state.get("mcts_stages", [])
        if state["current_step"] in mcts_stages and not state.get("mcts_completed", False):
            return "mcts_stage"
            
        # それ以外（成功かつMCTS不要）なら終了
        return "end_success"

    workflow.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "diagnoser": "diagnoser",
            "mcts_stage": "mcts_stage",
            "end_success": END
        }
    )

    # 4. MCTS探索後のルーティング
    workflow.add_conditional_edges(
        "mcts_stage",
        lambda state: "diagnoser" if state.get("status") == "FAILED" else "end_success",
        {
            "diagnoser": "diagnoser",
            "end_success": END
        }
    )

    # 5. Diagnoser から Replanner へ
    workflow.add_edge("diagnoser", "replanner")

    # 6. Replanner の後のルーティング（リトライループ）
    workflow.add_conditional_edges(
        "replanner",
        lambda state: "retry" if state.get("attempt_count", 0) < state.get("max_attempts", 3) else "end_fail",
        {
            "retry": "executor",
            "end_fail": END
        }
    )

    return workflow.compile()

# main.py から参照されるグローバル変数
agent_app = build_graph()
