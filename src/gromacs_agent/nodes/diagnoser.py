# src/gromacs_agent/nodes/diagnoser.py
import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import AuthenticationError, APIConnectionError, RateLimitError
from gromacs_agent.knowledge.db import KnowledgeBase
from gromacs_agent.core.state import AgentState

def diagnose_node(state: AgentState) -> dict:
    if state.get("status") != "FAILED":
        return {"status": "SUCCESS"}

    kb = KnowledgeBase()
    error_log = state.get("last_error") or ""
    similar_cases = kb.search(error_log)

    # APIキーがない場合のフォールバック
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {
            "diagnosis_context": {
                "cause": "Missing API Key (Dummy Diagnosis)",
                "fix_type": "PARAMETER_CHANGE",
                "parameters": {"dt": 0.001}
            },
            "status": "NEEDS_REPLAN"
        }

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
    except (AuthenticationError, APIConnectionError, RateLimitError, json.JSONDecodeError) as e:
        # APIキーが無効な場合や接続エラー、パースエラーの場合はダミーを返す
        diagnosis = {
            "cause": f"API Error or Parse Error: {str(e)}",
            "fix_type": "PARAMETER_CHANGE",
            "parameters": {"dt": 0.001}
        }

    return {
        "diagnosis_context": diagnosis,
        "status": "NEEDS_REPLAN"
    }
