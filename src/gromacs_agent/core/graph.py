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


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", plan_node)
    workflow.add_node("executor", execute_node)
    workflow.add_node("mcts_stage", mcts_stage_node)
    workflow.add_node("diagnoser", diagnose_node)
    workflow.add_node("replanner", replan_node)
    workflow.add_node("advance_stage", advance_stage_node)

    workflow.set_entry_point("planner")

    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"mcts": "mcts_stage", "execute": "executor"},
    )

    workflow.add_conditional_edges(
        "executor",
        _route_after_executor,
        {"advance": "advance_stage", "diagnose": "diagnoser"},
    )

    workflow.add_conditional_edges(
        "mcts_stage",
        _route_after_mcts,
        {"advance": "advance_stage", "end_fail": END},
    )

    workflow.add_edge("diagnoser", "replanner")

    workflow.add_conditional_edges(
        "replanner",
        _route_after_replanner,
        {"retry": "executor", "end_fail": END},
    )

    workflow.add_conditional_edges(
        "advance_stage",
        lambda s: "done" if s.get("status") == "ALL_DONE" else "next",
        {"done": END, "next": "planner"},
    )

    return workflow.compile()
