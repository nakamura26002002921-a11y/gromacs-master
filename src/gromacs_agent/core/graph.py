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
    workflow.add_node("advance_stage", advance_stage_node)
    workflow.add_node("stage_failed", stage_failed_node)

    # 2. エントリーポイントと最初のエッジ
    workflow.set_entry_point("planner")
    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "mcts": "mcts_stage",
            "execute": "executor",
        }
    )

    # 3. executor の後のルーティングロジック
    def route_after_executor(state):
        # 辞書アクセス (.get) を使用
        if state.get("last_error") is not None:
            return "diagnoser"

        # 成功したら次のステージへ進める (advance_stageがworkflow終端かどうかも判定する)
        return "advance"

    workflow.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "diagnoser": "diagnoser",
            "advance": "advance_stage",
        }
    )

    # 4. MCTS探索後のルーティング
    workflow.add_conditional_edges(
        "mcts_stage",
        lambda state: "diagnoser" if state.get("last_error") is not None else "advance",
        {
            "diagnoser": "diagnoser",
            "advance": "advance_stage",
        }
    )

    # 5. Diagnoser から Replanner へ
    workflow.add_edge("diagnoser", "replanner")

    # 6. Replanner の後のルーティング（リトライループ）
    workflow.add_conditional_edges(
        "replanner",
        lambda state: "retry" if (state.get("attempt_count", 0) < state.get("max_attempts", 3)) else "end_fail",
        {
            "retry": "executor",
            "end_fail": "stage_failed"
        }
    )

    # 7. advance_stage の後のルーティング（全ステージ完了 or 次ステージへ）
    workflow.add_conditional_edges(
        "advance_stage",
        lambda state: "done" if state.get("status") == "ALL_DONE" else _route_after_planner(state),
        {
            "done": END,
            "mcts": "mcts_stage",
            "execute": "executor",
        }
    )

    # 8. stage_failed は終端
    workflow.add_edge("stage_failed", END)

    return workflow.compile()

# main.py から参照されるグローバル変数
agent_app = build_graph()
