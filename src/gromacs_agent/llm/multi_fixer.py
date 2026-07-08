# src/gromacs_agent/llm/multi_fixer.py
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

def generate_multiple_fixes(stage: str, current_args: list, stderr: str, kb_suggestions: list) -> list:
    """
    LLMに複数の修正候補を生成させる。
    戻り値: [{"args": [...], "reason": "...", "confidence": 0.9}, ...]
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7) # 多様性を出すためtemperatureを上げる
    
    kb_context = ""
    if kb_suggestions:
        kb_context = "Past successful fixes for similar errors:\n" + \
                     "\n".join([f"- {s['reason']}: {s['fixed_args']}" for s in kb_suggestions])

    prompt = ChatPromptTemplate.from_template(
        """You are an expert in GROMACS. A command failed.
**Stage:** {stage}
**Original Args:** {args}
**Error:** {stderr}

{kb_context}

Generate exactly 3 DIFFERENT and PLAUSIBLE strategies to fix this error. 
Consider different physical or topological reasons (e.g., changing water model, adding -ignh, adjusting mdp parameters like emstep or nsteps).

Return ONLY a JSON list of objects:
[
  {{"args": ["new", "args"], "reason": "Brief reason", "confidence": 0.9}},
  {{"args": ["another", "fix"], "reason": "Brief reason", "confidence": 0.7}},
  {{"args": ["third", "option"], "reason": "Brief reason", "confidence": 0.5}}
]
If no fix is possible, return an empty list [].
"""
    )
    
    chain = prompt | llm
    response = chain.invoke({
        "stage": stage, "args": current_args, 
        "stderr": stderr[-1500:], "kb_context": kb_context
    })
    
    try:
        content = response.content.strip()
        if content.startswith("```json"): content = content[7:-3]
        elif content.startswith("```"): content = content[3:-3]
        return json.loads(content)
    except Exception as e:
        print(f"LLM JSON parse error: {e}")
        return []
