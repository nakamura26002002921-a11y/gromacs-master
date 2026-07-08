# src/gromacs_agent/nodes/executor.py
import os
import tempfile
from gromacs_agent.tools.gromacs_tools import GromacsTools
from gromacs_agent.core.state import AgentState


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


def _run_grompp(stage: str, mdp_file: str) -> tuple[int, str, str]:
    """gromppを実行してTPRファイルを生成"""
    tools = GromacsTools()
    args = [
        "-f", mdp_file,
        "-c", "processed.gro",
        "-p", "topol.top",
        "-o", f"{stage}.tpr"
    ]
    return tools.run_gmx_command("grompp", args)


def _run_mdrun(stage: str) -> tuple[int, str, str]:
    """mdrunを実行"""
    tools = GromacsTools()
    args = ["-deffnm", stage, "-s", f"{stage}.tpr"]
    return tools.run_gmx_command("mdrun", args)


def execute_node(state: AgentState) -> dict:
    step = state["current_step"]
    config = state.get("current_config", {})
    tools = GromacsTools()

    # MCTS対象外の初期ステージ（pdb2gmx, editconf等）は単一コマンド
    simple_stages = {
        "pdb2gmx": ["-f", "input.pdb", "-o", "processed.gro", "-water", "tip3p", "-ff", "amber99sb-ildn"],
        "editconf": ["-c", "processed.gro", "-o", "box.gro", "-d", "1.0", "-bt", "cubic"],
        "solvate": ["-cp", "box.gro", "-cs", "spc216.gro", "-o", "solvated.gro", "-p", "topol.top"],
        "genion": ["-s", "ions.tpr", "-o", "neutral.gro", "-p", "topol.top", "-pname", "NA", "-nname", "CL", "-neutral"],
    }

    if step in simple_stages:
        code, stdout, stderr = tools.run_gmx_command(step, simple_stages[step])
        return {
            "status": "SUCCESS" if code == 0 else "FAILED",
            "last_error": stderr if code != 0 else None,
            "log_snippet": stderr[-1000:] if code != 0 else None,
        }

    # em, nvt, npt, md は grompp → mdrun の2段階
    # 一時的なMDPファイルを作成
    mdp_content = _build_mdp_content(step, config)
    mdp_file = f"{step}.mdp"
    
    try:
        with open(mdp_file, "w") as f:
            f.write(mdp_content)
    except Exception as e:
        return {
            "status": "FAILED",
            "last_error": f"Failed to write MDP file: {str(e)}",
            "log_snippet": "",
        }

    # 1. grompp
    code, stdout, stderr = _run_grompp(step, mdp_file)
    if code != 0:
        return {
            "status": "FAILED",
            "last_error": f"grompp failed: {stderr}",
            "log_snippet": stderr[-1000:],
        }

    # 2. mdrun
    code, stdout, stderr = _run_mdrun(step)
    return {
        "status": "SUCCESS" if code == 0 else "FAILED",
        "last_error": stderr if code != 0 else None,
        "log_snippet": stderr[-1000:] if code != 0 else None,
    }
