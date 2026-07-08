# src/gromacs_agent/tools/gromacs_tools.py
import os
import re
import subprocess
from typing import Dict, Tuple
import structlog

from gromacs_agent.utils.command_logger import log_command

logger = structlog.get_logger()


class GromacsTools:
    @staticmethod
    def run_gmx_command(cmd: str, args: list, cwd: str = None) -> Tuple[int, str, str]:
        """
        Execute a GROMACS command.

        Args:
            cmd: GROMACS command (e.g., 'grompp', 'mdrun')
            args: List of arguments
            cwd: Working directory. コマンドの実行と reproduce.sh への記録の両方に使う。
                 Noneの場合はカレントディレクトリ (呼び出し側は原則 work_dir を明示すること)。

        Returns: (return_code, stdout, stderr)
        """
        full_cmd = ["gmx", cmd] + args
        work_dir = cwd if cwd else os.getcwd()

        # 実行したコマンドをbashスクリプトとして記録 (消えても再現できるようにする)
        log_command(work_dir, full_cmd)

        logger.info("Executing GROMACS command", cmd=full_cmd, cwd=work_dir)

        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                cwd=work_dir,
                timeout=3600,  # 1 hour timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except FileNotFoundError:
            return -1, "", "GROMACS not found in PATH"

    @staticmethod
    def partial_reward(log: str) -> float:
        """
        失敗時でも「どこまで進んだか」から部分的な報酬を与えるヒューリスティック (MCTS用)。
        ログから "Step" の到達数が読み取れない場合は 0.0 とする。
        """
        if not log:
            return 0.0
        matches = re.findall(r"Step\s+(\d+)", log)
        if not matches:
            return 0.0
        reached = int(matches[-1])
        target_matches = re.findall(r"nsteps\s*=\s*(\d+)", log)
        target = int(target_matches[-1]) if target_matches else max(reached, 1)
        return max(0.0, min(0.3, (reached / target) * 0.3))
