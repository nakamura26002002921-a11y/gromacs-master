# src/gromacs_agent/nodes/diagnoser.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from gromacs_agent.knowledge.db import KnowledgeBase
from gromacs_agent.core.state import AgentState
import json

def diagnose_node(state: AgentState) -> dict:
    if state["status"] != "FAILED":
        return {"status": "SUCCESS"}

    kb = KnowledgeBase()
    error_log = state["last_error"] or ""
    
    # RAGによる検索
    similar_cases = kb.search(error_log)
    
    # LLMによる診断
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
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | llm
    response = chain.invoke({"log": error_log, "cases": json.dumps(similar_cases)})
    
    # 実際のアプリではPydanticでパースするが、ここでは簡略化
    diagnosis = json.loads(response.content)
    
    return {
        "diagnosis_context": diagnosis,
        "status": "NEEDS_REPLAN"
    }
