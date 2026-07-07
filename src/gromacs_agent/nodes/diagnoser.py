# src/gromacs_agent/nodes/diagnoser.py
import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAIError
from gromacs_agent.knowledge.db import KnowledgeBase
from gromacs_agent.core.state import AgentState


def _fallback_diagnosis(error_log: str, reason: str) -> dict:
    params = {}
    log_lower = (error_log or "").lower()
    if "lincs" in log_lower:
        params = {"dt": 0.001}
    elif "blowing up" in log_lower or "instability" in log_lower:
        params = {"emtol": 500.0, "nsteps": 100000}
    elif "segmentation fault" in log_lower:
        params = {"dt": 0.001}
    else:
        params = {"dt": 0.001}

    return {
        "diagnosis_context": {
            "cause": f"Fallback diagnosis ({reason})",
            "fix_type": "PARAMETER_CHANGE",
            "parameters": params,
        },
        "status": "NEEDS_REPLAN",
    }


def diagnose_node(state: AgentState) -> dict:
    if state.get("status") != "FAILED":
        return {"status": "SUCCESS"}

    kb = KnowledgeBase()
    error_log = state.get("last_error") or ""
    similar_cases = kb.search(error_log)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _fallback_diagnosis(error_log, "no API key")

    prompt = ChatPromptTemplate.from_template("""
    You are an expert Computational Chemist.
    Analyze the following GROMACS error log and suggest the root cause and fix.

    Error Log:
    {log}

    Similar Past Cases (Knowledge Base):
    {cases}

    Output JSON format:
    {{"cause": "string", "fix_type": "PARAMETER_CHANGE" | "WORKFLOW_CHANGE", "parameters": {{}} }}
    """)

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        chain = prompt | llm
        response = chain.invoke({"log": error_log, "cases": json.dumps(similar_cases)})
        diagnosis = json.loads(response.content)
    except OpenAIError as e:
        return _fallback_diagnosis(error_log, f"OpenAI error: {e}")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return _fallback_diagnosis(error_log, f"Parse error: {e}")

    return {
        "diagnosis_context": diagnosis,
        "status": "NEEDS_REPLAN",
    }
