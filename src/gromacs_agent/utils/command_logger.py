# src/gromacs_agent/utils/command_logger.py
import os
import shlex

def log_bash_command(work_dir: str, cmd: list, script_name: str = "reproduce.sh"):
    """
    実行したコマンドをbashスクリプトとしてワークディレクトリに追記する。
    
    Args:
        work_dir (str): コマンドを実行したディレクトリ
        cmd (list): 実行したコマンドのリスト (例: ["gmx", "grompp", "-f", "topol.top", ...])
        script_name (str): 出力するスクリプト名
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
            
    # コマンドリストを文字列に変換（shlex.quoteで安全にエスケープ）
    escaped_cmd = " ".join([shlex.quote(str(c)) for c in cmd])
    
    # cd コマンドと、実行コマンドを追記
    with open(script_path, "a") as f:
        f.write(f"cd {shlex.quote(work_dir)}\n")
        f.write(f"{escaped_cmd}\n\n")
