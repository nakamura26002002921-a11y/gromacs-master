import os
import json
import structlog
from typing import List, Optional

# LangChain のインポート
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from gromacs_agent.tools.gromacs_tools import GromacsTools
from gromacs_agent.core.state import AgentState

logger = structlog.get_logger()

# ==========================================================
# LLM によるエラー解析と修正コマンドの生成
# ==========================================================
def get_llm_fix(step: str, current_args: List[str], stderr: str) -> Optional[List[str]]:
    """
    エラーメッセージをLLMに渡して解析させ、修正後の引数リストを生成する。
    """
    try:
        # APIキーが設定されていない場合はスキップ
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not set. Skipping LLM fix.")
            return None

        # gpt-4o-mini はコストが安く、構造化データの出力に優れている
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        prompt = ChatPromptTemplate.from_template(
            """You are an expert in GROMACS molecular dynamics simulations.
A GROMACS command has failed. Your task is to fix the command arguments to resolve the error.

**Command Stage:** {step}
**Original Arguments:** {args}
**Error Message (stderr):**
{stderr}

**Instructions:**
1. Analyze the error message carefully.
2. Determine the necessary changes to the arguments (e.g., adding flags like `-ignh`, changing water models, adjusting box sizes, or using `-v` for verbose).
3. Return a **complete, valid list of arguments** for the `{step}` command.
4. Do NOT include the command name itself (e.g., "pdb2gmx") in the list, only the arguments.
5. If the error is unfixable by changing arguments (e.g., missing input file), return `null`.

**Output Format:**
Return ONLY a valid JSON object with the following structure, no markdown or extra text:
{{
  "fixed_args": ["arg1", "arg2", ...],
  "reason": "Brief explanation of the fix"
}}
If unfixable, return:
{{
  "fixed_args": null,
  "reason": "Reason why it cannot be fixed"
}}
"""
        )
        
        chain = prompt | llm
        # エラーメッセージが長すぎる場合は末尾のみ渡す
        error_snippet = stderr[-1500:] if len(stderr) > 1500 else stderr
        
        response = chain.invoke({
            "step": step,
            "args": current_args,
            "stderr": error_snippet
        })
        
        content = response.content.strip()
        # LLMがマークダウンコードブロックを出力する場合があるため除去
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        data = json.loads(content)
        
        if data.get("fixed_args") is not None:
            logger.info("LLM proposed fix", reason=data.get("reason"), new_args=data["fixed_args"])
            return data["fixed_args"]
        else:
            logger.warning("LLM determined error is unfixable", reason=data.get("reason"))
            return None

    except Exception as e:
        logger.error("LLM fix attempt failed", error=str(e))
        return None


# ==========================================================
# メインのノード処理
# ==========================================================
def execute_node(state: AgentState) -> dict:
    step = state.get("current_step")
    raw_config = state.get("current_config", {})
    config = raw_config.get("stage_overrides", {}).get(step, raw_config)

    work_dir = state.get("work_dir") or os.getcwd()
    os.makedirs(work_dir, exist_ok=True)
    tools = GromacsTools()

    pdb_file = state.get("pdb_file", "input.pdb")
    force_field = config.get("force_field", "amber99sb-ildn")
    water_model = config.get("water", config.get("water_model", "tip3p"))

    # --- editconf の引数を config から動的に構築 ---
    editconf_args = ["-f", "processed.gro", "-o", "box.gro", "-c"]
    box_size = config.get("box_size")
    box_distance = config.get("box_distance")
    box_type = config.get("box_type", "cubic")

    if box_size:
        editconf_args += ["-box"] + box_size.split()
    else:
        distance = box_distance if box_distance is not None else 1.0
        editconf_args += ["-d", str(distance)]
    editconf_args += ["-bt", box_type]

    # 1. 軽量ステージ (pdb2gmx, editconf, solvate)
    simple_stages = {
        "pdb2gmx": ["-f", pdb_file, "-o", "processed.gro", "-water", water_model, "-ff", force_field, "-ignh"],
        "editconf": editconf_args,
        "solvate": ["-cp", "box.gro", "-cs", "spc216.gro", "-o", "solvated.gro", "-p", "topol.top"],
    }

    if step in simple_stages:
        args = simple_stages[step]
        
        # 🚀 LLMによる自動修復ループ (最大3回)
        max_retries = 3
        for attempt in range(max_retries):
            code, stdout, stderr = tools.run_gmx_command(step, args, cwd=work_dir)
            
            if code == 0:
                return {
                    "status": "SUCCESS",
                    "last_error": None,
                    "log_snippet": None,
                    "work_dir": work_dir,
                }
            
            logger.warning(f"Command failed (attempt {attempt+1}/{max_retries}). Asking LLM for fix...", step=step)
            
            # LLMにエラーを解析させ、修正後の引数を取得
            fixed_args = get_llm_fix(step, args, stderr)
            
            if fixed_args is None or fixed_args == args:
                logger.error("LLM could not provide a valid fix or no change was made.", step=step)
                break
                
            # 修正後の引数で再試行
            args = fixed_args
            
        # すべて失敗した場合
        return {
            "status": "FAILED",
            "last_error": stderr,
            "log_snippet": stderr[-1000:],
            "work_dir": work_dir,
        }

    # 2. MDP生成と grompp/mdrun (em, nvt, npt, md)
    # ... (既存の grompp/mdrun ロジックをここに記述) ...
    # ※ 必要に応じて、ここにも同様のLLM修復ループを組み込んでください
