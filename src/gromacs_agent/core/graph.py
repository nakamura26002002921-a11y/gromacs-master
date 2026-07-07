# src/gromacs_agent/core/graph.py
from langgraph.graph import StateGraph, END
from gromacs_agent.core.state import AgentState
from gromacs_agent.nodes.planner import plan_node
from gromacs_agent.nodes.executor import execute_node
from gromacs_agent.nodes.diagnoser import diagnose_node
from gromacs_agent.nodes.replanner import replan_node

def route_after_executor(state: AgentState) -> str:
    """実行後の分岐ロジック"""
    if state.get("status") == "FAILED":
        return "diagnose"
    # 成功時は次のステップへ（簡易化のためここではENDとする）
    # 本来は state["step_index"] を進めて execute に戻すループを作る
    return "end"

def route_after_replanner(state: AgentState) -> str:
    """修正後の分岐ロジック"""
    if state.get("attempt_count", 0) >= state.get("max_attempts", 3):
        return "end"
    return "retry"

def build_graph():
    workflow = StateGraph(AgentState)

    # ノード追加
    workflow.add_node("planner", plan_node)
    workflow.add_node("executor", execute_node)
    workflow.add_node("diagnoser", diagnose_node)
    workflow.add_node("replanner", replan_node)

    # エッジ定義
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    
    # 条件付きエッジ (修正版)
    workflow.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "diagnose": "diagnoser",
            "end": END
        }
    )
    
    workflow.add_edge("diagnoser", "replanner")
    
    workflow.add_conditional_edges(
        "replanner",
        route_after_replanner,
        {
            "retry": "executor",
            "end": END
        }
    )

    return workflow.compile()

# インポートエラーを防ぐためのインスタンス化
try:
    agent_app = build_graph()
except Exception as e:
    print(f"Graph build error: {e}")
    agent_app = None
