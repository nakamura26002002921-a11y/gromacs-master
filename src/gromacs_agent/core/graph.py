# src/gromacs_agent/core/graph.py
from langgraph.graph import StateGraph, END
from gromacs_agent.core.state import AgentState
from gromacs_agent.nodes.planner import plan_node
from gromacs_agent.nodes.executor import execute_node
from gromacs_agent.nodes.diagnoser import diagnose_node
from gromacs_agent.nodes.replanner import replan_node
from gromacs_agent.nodes.mcts_stage import mcts_stage_node

def route_stage(state: AgentState) -> str:
    """現在のステージがMCTS対象かどうかでルーティング"""
    # mcts_stages リストに現在のステップが含まれているか確認
    if state["current_step"] in state.get("mcts_stages", []):
        return "mcts"
    return "execute"

def after_execution(state: AgentState) -> str:
    """実行後の分岐: 成功なら次ステージへ、失敗なら診断へ"""
    if state.get("status") == "SUCCESS":
        return "advance"
    return "diagnose"

def after_mcts(state: AgentState) -> str:
    """MCTS後の分岐: 成功なら次ステージへ、失敗なら終了"""
    if state.get("status") == "SUCCESS":
        return "advance"
    return "end_fail"

def advance_stage_node(state: AgentState) -> dict:
    """次のステージへ進める、あるいは完了"""
    next_index = state["step_index"] + 1
    workflow = state["workflow"]
    
    # 全ステージ完了
    if next_index >= len(workflow):
        return {"status": "ALL_DONE", "step_index": next_index}
    
    # 次のステージへ
    return {
        "step_index": next_index,
        "current_step": workflow[next_index],
        "status": "PENDING",
        "attempt_count": 0, # リトライカウントをリセット
        "last_error": None
    }

def build_graph():
    workflow = StateGraph(AgentState)

    # ノード追加
    workflow.add_node("planner", plan_node)
    workflow.add_node("executor", execute_node)
    workflow.add_node("mcts_stage", mcts_stage_node)
    workflow.add_node("diagnoser", diagnose_node)
    workflow.add_node("replanner", replan_node)
    workflow.add_node("advance_stage", advance_stage_node)

    # エッジ定義
    workflow.set_entry_point("planner")

    # 1. Planner -> (MCTS対象か？) -> mcts_stage / executor
    workflow.add_conditional_edges(
        "planner",
        route_stage,
        {
            "mcts": "mcts_stage",
            "execute": "executor"
        }
    )

    # 2. Executor -> (成功？) -> advance / diagnose
    workflow.add_conditional_edges(
        "executor",
        after_execution,
        {
            "advance": "advance_stage",
            "diagnose": "diagnoser"
        }
    )

    # 3. MCTS Stage -> (成功？) -> advance / end
    workflow.add_conditional_edges(
        "mcts_stage",
        after_mcts,
        {
            "advance": "advance_stage",
            "end_fail": END
        }
    )

    # 4. Diagnoser -> Replanner
    workflow.add_edge("diagnoser", "replanner")

    # 5. Replanner -> (リトライ可能？) -> executor / end
    workflow.add_conditional_edges(
        "replanner",
        lambda state: "retry" if state.get("attempt_count", 0) < state.get("max_attempts", 3) else "end_fail",
        {
            "retry": "executor",
            "end_fail": END
        }
    )

    # 6. Advance Stage -> (完了？) -> end / (次のステージへ) -> route_stage
    workflow.add_conditional_edges(
        "advance_stage",
        lambda state: "done" if state.get("status") == "ALL_DONE" else "next",
        {
            "done": END,
            "next": "planner" # 次のステージの準備のためplannerに戻る（または直接route_stageへ戻すよう変更可能）
        }
    )

    return workflow.compile()

# インポート時にグラフをビルド
agent_app = build_graph()
