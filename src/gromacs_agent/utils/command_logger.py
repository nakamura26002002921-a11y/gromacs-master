# src/gromacs_agent/utils/command_logger.py
"""
実行したgmxコマンドと、動的に生成したファイル(.mdp等)の中身を
work_dir内の1本のbashスクリプト(既定: reproduce.sh)に書き起こすモジュール。

目的: work_dir以下の中間ファイルが全て消えても、この reproduce.sh と
最初の入力ファイル (pdbファイル等) さえあれば、gmx_agentを介さずに
シェルだけで全く同じ手順を再現できるようにすること。

このモジュールが唯一の実装であり、gromacs_tools.py 等はこれをimportして使う
(以前 gromacs_tools.py 内に同等のロジックを重複実装していたが、一本化した)。
"""
import os
import shlex

DEFAULT_SCRIPT_NAME = "reproduce.sh"


def _ensure_header(script_path: str):
    if not os.path.exists(script_path):
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("set -e  # Exit immediately if a command exits with a non-zero status\n")
            f.write("# Auto-generated reproduction script by gromacs_agent\n")
            f.write("# work_dir内のファイルが全て消えても、このスクリプトと最初の入力ファイル\n")
            f.write("# だけで全く同じ手順を再現できます。\n\n")
        os.chmod(script_path, 0o755)


def log_command(work_dir: str, cmd: list, script_name: str = DEFAULT_SCRIPT_NAME):
    """
    実行したコマンド (例: ["gmx", "grompp", "-f", "em.mdp", ...]) を追記する。
    """
    if not work_dir or not cmd:
        return

    script_path = os.path.join(work_dir, script_name)
    _ensure_header(script_path)

    escaped_cmd = " ".join(shlex.quote(str(c)) for c in cmd)
    with open(script_path, "a") as f:
        f.write(f"{escaped_cmd}\n")


def log_file_write(work_dir: str, filename: str, content: str, script_name: str = DEFAULT_SCRIPT_NAME):
    """
    Pythonから動的に生成したファイル (.mdp等) の中身を、ヒアドキュメントとして
    bashスクリプトに埋め込む。これにより、生成ロジックがPython側にしか無い
    ファイルも reproduce.sh 単体で再現できる。
    """
    if not work_dir or not filename:
        return

    script_path = os.path.join(work_dir, script_name)
    _ensure_header(script_path)

    # 中身にEOFという文字列が含まれる衝突を避けるため一意な区切り文字を使う
    delimiter = "GMX_AGENT_EOF"
    with open(script_path, "a") as f:
        f.write(f"cat > {shlex.quote(filename)} << '{delimiter}'\n")
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")
        f.write(f"{delimiter}\n")


def write_file_and_log(work_dir: str, filename: str, content: str, script_name: str = DEFAULT_SCRIPT_NAME):
    """
    ファイルを実際にwork_dir内へ書き込み、同時に reproduce.sh へも
    再現可能な形で記録する。実行ノードはこの関数を使う。
    """
    os.makedirs(work_dir, exist_ok=True)
    filepath = os.path.join(work_dir, filename)
    with open(filepath, "w") as f:
        f.write(content)
    log_file_write(work_dir, filename, content, script_name=script_name)
    return filepath


def reset_script(work_dir: str, script_name: str = DEFAULT_SCRIPT_NAME):
    """新しいrun開始時に、前回分が残らないようスクリプトを削除しておく。"""
    if not work_dir:
        return
    script_path = os.path.join(work_dir, script_name)
    if os.path.exists(script_path):
        os.remove(script_path)
