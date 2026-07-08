# main.py
"""
GROMACS Autonomous Agent エントリポイント。

使い方:
    python main.py                     # デフォルト (1ALC) を実行
    python main.py --pdb-id 1AKI       # 別のPDB IDを指定
    python main.py --pdb-file my.pdb   # ローカルのPDBファイルを使う (ダウンロードしない)
    python main.py --work-dir ./run1   # 作業ディレクトリを固定 (省略時は一時ディレクトリ)

注意: このスクリプトは実際に `gmx` コマンドを呼び出します。GROMACSが
PATHに通った環境で実行してください。
"""
import argparse
import os
import shutil
import sys
import tempfile
import urllib.request

from gromacs_agent.core.graph import build_graph
from gromacs_agent.utils.logger import setup_logger

RCSB_URL_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.pdb"


def _download_pdb(pdb_id: str, dest_path: str):
    url = RCSB_URL_TEMPLATE.format(pdb_id=pdb_id.upper())
    print(f"📥 Downloading {pdb_id} from {url} ...")
    try:
        urllib.request.urlretrieve(url, dest_path)
    except Exception as e:
        print(f"❌ Failed to download {pdb_id}.pdb: {e}", file=sys.stderr)
        print("   https://files.rcsb.org にネットワークアクセスできるか確認してください。", file=sys.stderr)
        sys.exit(1)


def build_initial_state(pdb_id: str, pdb_file: str, work_dir: str) -> dict:
    os.makedirs(work_dir, exist_ok=True)
    input_pdb_path = os.path.join(work_dir, "input.pdb")

    if pdb_file:
        shutil.copy(pdb_file, input_pdb_path)
        system_name = os.path.splitext(os.path.basename(pdb_file))[0]
    else:
        _download_pdb(pdb_id, input_pdb_path)
        system_name = pdb_id.upper()

    return {
        "system_name": system_name,
        "pdb_file": "input.pdb",  # work_dir内での相対パス (executor.pyのpdb2gmxが参照)
        "workflow": [],           # Plannerが生成
        "step_index": 0,
        "attempt_count": 0,
        "max_attempts": 3,
        "current_config": {
            "force_field": "amber99sb-ildn",
            "water": "tip3p",
            "dt": 0.002,
            "nsteps": 50000,
        },
        "history": [],
        # em/nvtなど「切りの良い単位」でMCTS探索を行うステージ
        "mcts_stages": ["em", "nvt"],
        "mcts_max_iterations": 4,
        "mcts_max_candidates": 3,
        "mcts_exploration_constant": 1.41,
        "work_dir": work_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="GROMACS Autonomous Agent")
    parser.add_argument("--pdb-id", default="1ALC", help="RCSBからダウンロードするPDB ID (デフォルト: 1ALC)")
    parser.add_argument("--pdb-file", default=None, help="ローカルのPDBファイルを使う場合のパス (指定時は--pdb-idを無視)")
    parser.add_argument("--work-dir", default=None, help="作業ディレクトリ (省略時は一時ディレクトリを自動生成)")
    args = parser.parse_args()

    setup_logger()

    default_name = (args.pdb_file and os.path.splitext(os.path.basename(args.pdb_file))[0]) or args.pdb_id
    work_dir = args.work_dir or tempfile.mkdtemp(prefix=f"gmx_agent_{default_name}_")

    print("🚀 Starting GROMACS Autonomous Agent...")
    print(f"📁 work_dir: {work_dir}")

    initial_state = build_initial_state(args.pdb_id, args.pdb_file, work_dir)

    agent_app = build_graph()
    result = agent_app.invoke(initial_state)

    print(f"✅ Final Status: {result['status']}")
    print(f"🔧 Final Config: {result['current_config']}")
    print(f"📜 History steps recorded: {len(result.get('history', []))}")
    print(f"📄 Reproduction script: {os.path.join(work_dir, 'reproduce.sh')}")


if __name__ == "__main__":
    main()
