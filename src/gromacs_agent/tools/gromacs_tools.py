# src/gromacs_agent/tools/gromacs_tools.py
import os
import shlex
import subprocess
from typing import Tuple
import structlog

logger = structlog.get_logger()

def _log_bash_command(work_dir: str, cmd: list, script_name: str = "reproduce.sh"):
    """
    実行したコマンドをbashスクリプトとしてワークディレクトリに追記する。
    """
    if not work_dir or not cmd:
        return
    
    script_path = os.path.join(work_dir, script_name)
    
    # ファイルが存在しない場合（初回実行時）は、bashのヘッダーを書き込む
    if not os.path.exists(script_path):
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("set -e  # Exit immediately if a command exits with a non-zero status\n")
            f.write("# Auto-generated reproduction script by gromacs_agent\n\n")
            
    # コマンドリストを文字列に変換（shlex.quoteでパスにスペースが含まれていても安全）
    escaped_cmd = " ".join([shlex.quote(str(c)) for c in cmd])
    
    # cd コマンドと、実行コマンドを追記
    with open(script_path, "a") as f:
        f.write(f"cd {shlex.quote(work_dir)}\n")
        f.write(f"{escaped_cmd}\n\n")


class GromacsTools:
    @staticmethod
    def run_gmx_command(cmd: str, args: list, cwd: str = None) -> Tuple[int, str, str]:
        """
        Execute a GROMACS command.
        
        Args:
            cmd: GROMACS command (e.g., 'grompp', 'mdrun')
            args: List of arguments
            cwd: Working directory. If None, uses current directory.
            
        Returns: (return_code, stdout, stderr)
        """
        full_cmd = ["gmx", cmd] + args
        
        # 作業ディレクトリの決定（指定がなければカレントディレクトリ）
        work_dir = cwd if cwd else os.getcwd()
        
        # ★追加: bashスクリプトに記録
        _log_bash_command(work_dir, full_cmd)
        
        logger.info("Executing GROMACS command", cmd=full_cmd, cwd=work_dir)
        
        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                cwd=work_dir,  # ★追加: 明示的にcwdを指定して実行
                timeout=3600   # 1 hour timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except FileNotFoundError:
            return -1, "", "GROMACS not found in PATH"
