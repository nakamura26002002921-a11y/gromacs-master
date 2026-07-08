# src/gromacs_agent/nodes/executor.py
import os
from gromacs_agent.tools.gromacs_tools import GromacsTools
from gromacs_agent.core.state import AgentState
from gromacs_agent.utils.command_logger import write_file_and_log


def _build_mdp_content(step: str, config: dict) -> str:
    """ステージに応じた最小限のMDPファイル内容を生成"""
    dt = config.get("dt", 0.002)
    nsteps = config.get("nsteps", 50000)
    emtol = config.get("emtol", 1000.0)

    if step == "em":
        return f"integrator = steep\nemtol = {emtol}\nnsteps = {nsteps}\n"
    elif step in ("nvt", "npt", "md"):
        return f"integrator = md\ndt = {dt}\nnsteps = {nsteps}\n"
    return ""


# ステージごとに「grompp -c」へ渡すべき直前ステージの出力ファイル
_COORD_IN = {
    "em": "neutral.gro",   # genionの出力 (中性化された溶媒和済み構造)
    "nvt": "em.gro",       # emの出力
    "npt": "nvt.gro",      # nvtの出力
    "md": "npt.gro",       # nptの出力
}


def _run_grompp(stage: str, mdp_file: str, work_dir: str) -> tuple[int, str, str]:
    """gromppを実行してTPRファイルを生成。-cには直前ステージの出力を正しく連鎖させる。"""
    tools = GromacsTools()
    args = [
        "-f", mdp_file,
        "-c", _COORD_IN.get(stage, "processed.gro"),
        "-p", "topol.top",
        "-o", f"{stage}.tpr",
    ]
    return tools.run_gmx_command("grompp", args, cwd=work_dir)


def _run_mdrun(stage: str, work_dir: str) -> tuple[int, str, str]:
    """mdrunを実行"""
    tools = GromacsTools()
    args = ["-deffnm", stage, "-s", f"{stage}.tpr"]
    return tools.run_gmx_command("mdrun", args, cwd=work_dir)


def execute_node(state: AgentState) -> dict:
    step = state["current_step"]
    config = state.get("current_config", {})
    # work_dirが未指定ならカレントディレクトリにフォールバック
    # (本来は main.py / planner が tempfile.mkdtemp() 等で用意し、state に積んでおく)
    work_dir = state.get("work_dir") or os.getcwd()
    os.makedirs(work_dir, exist_ok=True)
    tools = GromacsTools()

    pdb_file = state.get("pdb_file", "input.pdb")
    force_field = config.get("force_field", "amber99sb-ildn")
    water_model = config.get("water", config.get("water_model", "tip3p"))

    # MCTS対象外の初期ステージ（pdb2gmx, editconf等）は単一コマンド
    simple_stages = {
        "pdb2gmx": ["-f", pdb_file, "-o", "processed.gro", "-water", water_model, "-ff", force_field],
        "editconf": ["-f", "processed.gro", "-o", "box.gro", "-c", "-d", "1.0", "-bt", "cubic"],
        "solvate": ["-cp", "box.gro", "-cs", "spc216.gro", "-o", "solvated.gro", "-p", "topol.top"],
    }

    if step == "genion":
        # genionは事前に ions.tpr (solvate後の構造に対するgrompp出力) が必要。
        # ions.mdpは空実行用の最小構成でよい (実際のイオン付加パラメータはgenionの引数側で指定する)。
        try:
            write_file_and_log(work_dir, "ions.mdp", "integrator = steep\nnsteps = 0\n")
        except Exception as e:
            return {
                "status": "FAILED",
                "last_error": f"Failed to write ions.mdp: {str(e)}",
                "log_snippet": "",
                "work_dir": work_dir,
            }

        code, stdout, stderr = tools.run_gmx_command(
            "grompp",
            ["-f", "ions.mdp", "-c", "solvated.gro", "-p", "topol.top", "-o", "ions.tpr"],
            cwd=work_dir,
        )
        if code != 0:
            return {
                "status": "FAILED",
                "last_error": f"grompp (ions.tpr) failed: {stderr}",
                "log_snippet": stderr[-1000:],
                "work_dir": work_dir,
            }

        code, stdout, stderr = tools.run_gmx_command(
            "genion",
            ["-s", "ions.tpr", "-o", "neutral.gro", "-p", "topol.top", "-pname", "NA", "-nname", "CL", "-neutral"],
            cwd=work_dir,
        )
        return {
            "status": "SUCCESS" if code == 0 else "FAILED",
            "last_error": stderr if code != 0 else None,
            "log_snippet": stderr[-1000:] if code != 0 else None,
            "work_dir": work_dir,
        }

    if step in simple_stages:
        code, stdout, stderr = tools.run_gmx_command(step, simple_stages[step], cwd=work_dir)
        return {
            "status": "SUCCESS" if code == 0 else "FAILED",
            "last_error": stderr if code != 0 else None,
            "log_snippet": stderr[-1000:] if code != 0 else None,
            "work_dir": work_dir,
        }

    # em, nvt, npt, md は grompp → mdrun の2段階
    mdp_content = _build_mdp_content(step, config)
    mdp_file = f"{step}.mdp"

    try:
        # 実ファイルに書き込むと同時に、reproduce.sh へヒアドキュメントとして記録する。
        # これにより、この.mdpファイル自体が消えてもreproduce.sh単体で再現できる。
        write_file_and_log(work_dir, mdp_file, mdp_content)
    except Exception as e:
        return {
            "status": "FAILED",
            "last_error": f"Failed to write MDP file: {str(e)}",
            "log_snippet": "",
            "work_dir": work_dir,
        }

    # 1. grompp
    code, stdout, stderr = _run_grompp(step, mdp_file, work_dir)
    if code != 0:
        return {
            "status": "FAILED",
            "last_error": f"grompp failed: {stderr}",
            "log_snippet": stderr[-1000:],
            "work_dir": work_dir,
        }

    # 2. mdrun
    code, stdout, stderr = _run_mdrun(step, work_dir)
    return {
        "status": "SUCCESS" if code == 0 else "FAILED",
        "last_error": stderr if code != 0 else None,
        "log_snippet": stderr[-1000:] if code != 0 else None,
        "work_dir": work_dir,
    }
