# src/gromacs_agent/nodes/executor.py
import os
from gromacs_agent.tools.gromacs_tools import GromacsTools
from gromacs_agent.core.state import AgentState
from gromacs_agent.utils.command_logger import write_file_and_log


def _effective_config(step: str, config: dict) -> dict:
    """
    ステージ単位の上書き設定をマージした「実効config」を返す。

    current_config["stage_overrides"] = {
        "npt": {"pcoupl": "Berendsen"},
        "md":  {"pcoupl": "Parrinello-Rahman", "nsteps": 500000},
    }
    のように書くと、共通のベース設定 (dt, tcoupl, ref_t等) はそのままに、
    指定したステージだけ好きなパラメータを上書きできる。
    nvt→npt(Berendsen)の緩和計算は何も指定しなければ従来通りのデフォルトのまま変わらない。
    """
    base = {k: v for k, v in config.items() if k != "stage_overrides"}
    overrides = (config.get("stage_overrides") or {}).get(step, {})
    return {**base, **overrides}


def _build_mdp_content(step: str, config: dict) -> str:
    """
    ステージに応じたMDPファイル内容を生成する。configはすでに
    _effective_config() でステージ別上書きがマージされたものを渡すこと。

    configで指定できる主なキー:
        dt, nsteps, emtol                     : 全ステージ共通の基本パラメータ
        tcoupl, ref_t, tau_t                  : 温度制御 (nvt/npt/mdで使用)
        pcoupl, pcoupltype, ref_p, tau_p,
        compressibility                       : 圧力制御 (npt/mdで使用)
    """
    dt = config.get("dt", 0.002)
    nsteps = config.get("nsteps", 50000)
    emtol = config.get("emtol", 1000.0)

    tcoupl = config.get("tcoupl", "V-rescale")
    ref_t = config.get("ref_t", 300)
    tau_t = config.get("tau_t", 0.1)

    # 圧力制御のデフォルトはBerendsen (npt平衡化での標準的な選択)。
    # stage_overridesで md だけ Parrinello-Rahman に変える、といった使い方を想定。
    pcoupl = config.get("pcoupl", "Berendsen")
    pcoupltype = config.get("pcoupltype", "isotropic")
    ref_p = config.get("ref_p", 1.0)
    tau_p = config.get("tau_p", 2.0)
    compressibility = config.get("compressibility", 4.5e-5)

    if step == "em":
        return (
            f"integrator = steep\n"
            f"emtol = {emtol}\n"
            f"nsteps = {nsteps}\n"
        )

    if step == "nvt":
        # nvtでは圧力制御はまだ行わない (体積固定で温度だけ先に平衡化する)
        return (
            f"integrator = md\n"
            f"dt = {dt}\n"
            f"nsteps = {nsteps}\n"
            f"tcoupl = {tcoupl}\n"
            f"tc-grps = System\n"
            f"ref_t = {ref_t}\n"
            f"tau_t = {tau_t}\n"
            f"gen_vel = yes\n"
            f"gen_temp = {ref_t}\n"
            f"define = -DPOSRES\n"
        )

    if step in ("npt", "md"):
        lines = [
            "integrator = md",
            f"dt = {dt}",
            f"nsteps = {nsteps}",
            f"tcoupl = {tcoupl}",
            "tc-grps = System",
            f"ref_t = {ref_t}",
            f"tau_t = {tau_t}",
            f"pcoupl = {pcoupl}",
            f"pcoupltype = {pcoupltype}",
            f"ref_p = {ref_p}",
            f"tau_p = {tau_p}",
            f"compressibility = {compressibility}",
        ]
        if step == "npt":
            lines.append("define = -DPOSRES")  # nptもまだ拘束を残すのが標準的
            lines.append("gen_vel = no")        # nvtで得た速度を引き継ぐ
        # mdでは位置拘束を外し (define指定なし)、nvt/nptで得た速度をそのまま使う
        return "\n".join(lines) + "\n"

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
    raw_config = state.get("current_config", {})
    config = _effective_config(step, raw_config)

    work_dir = state.get("work_dir") or os.getcwd()
    os.makedirs(work_dir, exist_ok=True)
    tools = GromacsTools()

    pdb_file = state.get("pdb_file", "input.pdb")
    force_field = config.get("force_field", "amber99sb-ildn")
    water_model = config.get("water", config.get("water_model", "tip3p"))
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

    simple_stages = {
        "pdb2gmx": ["-f", pdb_file, "-o", "processed.gro", "-water", water_model, "-ff", force_field],
        "editconf": editconf_args,
        "solvate": ["-cp", "box.gro", "-cs", "spc216.gro", "-o", "solvated.gro", "-p", "topol.top"],
    }
    
    if step == "genion":
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
            stdin_input="SOL\n",
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
