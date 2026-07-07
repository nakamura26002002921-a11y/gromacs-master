# src/gromacs_agent/core/graph.py
from langgraph.graph import StateGraph, END
from gromacs_agent.core.state import AgentState
from gromacs_agent.nodes.planner import plan_node
from gromacs_agent.nodes.executor import execute_node
from gromacs_agent.nodes.diagnoser import diagnose_node
from gromacs_agent.nodes.replanner import replan_node

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

    # 【修正点】条件付きエッジの書き方をLangGraphの仕様に合わせる
    workflow.add_conditional_edges(
        "executor",
        lambda state: "diagnoser" if state.get("status") == "FAILED" else "end_success",
        {
            "diagnoser": "diagnoser",
            "end_success": END
        }
    )

    workflow.add_edge("diagnoser", "replanner")

    # リトライループ
    workflow.add_conditional_edges(
        "replanner",
        lambda state: "retry" if state.get("attempt_count", 0) < state.get("max_attempts", 3) else "end_fail",
        {
            "retry": "executor",
            "end_fail": END
        }
    )

    return workflow.compile()

# インポート時にグラフをビルド
agent_app = build_graph()
