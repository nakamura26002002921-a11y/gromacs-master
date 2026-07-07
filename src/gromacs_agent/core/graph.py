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
    
    # 条件付きエッジ: 成功なら次へ、失敗なら診断へ
    workflow.add_conditional_edges(
        "executor",
        lambda state: "diagnoser" if state["status"] == "FAILED" else "next_step"
    )
    
    workflow.add_edge("diagnoser", "replanner")
    
    # リトライループ
    workflow.add_conditional_edges(
        "replanner",
        lambda state: "executor" if state["attempt_count"] < state["max_attempts"] else END
    )

    return workflow.compile()

agent_app = build_graph()
