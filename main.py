# main.py
"""
GROMACS Autonomous Agent エントリポイント。

使い方:
    python main.py                     # デフォルト (1ALC) を実行
    python main.py --pdb-id 1AKI       # 別のPDB IDを指定
    python main.py --pdb-file my.pdb   # ローカルのPDBファイルを使う (ダウンロードしない)
    python main.py --work-dir ./run1   # 作業ディレクトリを固定 (省略時は一時ディレクトリ)
    python main.py --force-field charmm36-jul2022 --water tip4p

    # mdだけParrinello-Rahman + より長いnstepsにする (nvt/npt(Berendsen)はデフォルトのまま)
    python main.py --stage-override '{"md": {"pcoupl": "Parrinello-Rahman", "nsteps": 500000}}'

注意: このスクリプトは実際に `gmx` コマンドを呼び出します。GROMACSが
PATHに通った環境で実行してください。
"""
import argparse
import json
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


def build_initial_state(pdb_id: str, pdb_file: str, work_dir: str,
                         md_overrides: dict = None, stage_overrides: dict = None) -> dict:
    """
    current_config の構成:
        - トップレベルのキー (dt, nsteps, tcoupl, pcoupl等) は全ステージ共通のデフォルト。
        - current_config["stage_overrides"][<stage名>] に辞書を書くと、そのステージだけ
          好きなキーを上書きできる (例: {"md": {"pcoupl": "Parrinello-Rahman"}})。
          nvt → npt(Berendsen) の緩和計算部分は何も指定しなければ従来通り変わらない。
    """
    os.makedirs(work_dir, exist_ok=True)
    input_pdb_path = os.path.join(work_dir, "input.pdb")

    if pdb_file:
        shutil.copy(pdb_file, input_pdb_path)
        system_name = os.path.splitext(os.path.basename(pdb_file))[0]
    else:
        _download_pdb(pdb_id, input_pdb_path)
        system_name = pdb_id.upper()

    current_config = {
        "force_field": "amber99sb-ildn",
        "water": "tip3p",
        "dt": 0.002,
        "nsteps": 50000,
        # 温度制御 (nvt/npt/md共通のデフォルト)
        "tcoupl": "V-rescale",
        "ref_t": 300,
        "tau_t": 0.1,
        # 圧力制御 (npt/md共通のデフォルト)。npt平衡化はBerendsenが標準的。
        # 本番run (md) だけ変えたい場合は stage_overrides["md"] で上書きする。
        "pcoupl": "Berendsen",
        "pcoupltype": "isotropic",
        "ref_p": 1.0,
        "tau_p": 2.0,
        "compressibility": 4.5e-5,
        # ステージ別の自由な上書き設定 (デフォルトは空 = 全ステージ共通設定のみ)
        "stage_overrides": stage_overrides or {},
    }
    current_config.update({k: v for k, v in (md_overrides or {}).items() if v is not None})

    return {
        "system_name": system_name,
        "pdb_file": "input.pdb",  # work_dir内での相対パス (executor.pyのpdb2gmxが参照)
        "workflow": [],           # Plannerが生成
        "step_index": 0,
        "attempt_count": 0,
        "max_attempts": 3,
        "current_config": current_config,
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
    parser.add_argument("--force-field", default=None, help="pdb2gmxに渡す力場 (例: amber99sb-ildn, charmm36-jul2022)")
    parser.add_argument("--water", default=None, help="水モデル (例: tip3p, tip4p, spce)")
    parser.add_argument("--tcoupl", default=None, help="温度制御アルゴリズム (例: V-rescale, Berendsen, Nose-Hoover)")
    parser.add_argument("--ref-t", type=float, default=None, help="目標温度 [K]")
    parser.add_argument("--pcoupl", default=None, help="気圧制御アルゴリズム (全ステージ共通のデフォルトを変更)")
    parser.add_argument("--ref-p", type=float, default=None, help="目標圧力 [bar]")
    parser.add_argument("--tau-p", type=float, default=None, help="圧力の緩和時定数 [ps]")
    parser.add_argument(
        "--stage-override", default=None,
        help='ステージ別の上書き設定をJSONで指定 (例: \'{"md": {"pcoupl": "Parrinello-Rahman"}}\')',
    )
    parser.add_argument("--box-type", default=None,
                    help="ボックス形状 (cubic, dodecahedron, octahedron等)")
    parser.add_argument("--box-distance", type=float, default=None,
                    help="タンパク質からボックス境界までの距離 [nm] (例: 0.3 = 3Å)")
    parser.add_argument("--box-size", default=None,
                    help="ボックスの明示的なサイズ [nm] (例: '1.0 1.0 1.0' = 10Å立方体)")

    args = parser.parse_args()

    setup_logger()

    default_name = (args.pdb_file and os.path.splitext(os.path.basename(args.pdb_file))[0]) or args.pdb_id
    work_dir = args.work_dir or tempfile.mkdtemp(prefix=f"gmx_agent_{default_name}_")

    print("🚀 Starting GROMACS Autonomous Agent...")
    print(f"📁 work_dir: {work_dir}")

    stage_overrides = {}
    if args.stage_override:
        try:
            stage_overrides = json.loads(args.stage_override)
        except json.JSONDecodeError as e:
            print(f"❌ --stage-override のJSONが不正です: {e}", file=sys.stderr)
            sys.exit(1)

    md_overrides = {
        "force_field": args.force_field,
        "water": args.water,
        "tcoupl": args.tcoupl,
        "ref_t": args.ref_t,
        "pcoupl": args.pcoupl,
        "ref_p": args.ref_p,
        "tau_p": args.tau_p,
        "box_type": args.box_type,
        "box_distance": args.box_distance,
        "box_size": args.box_size,
    }
    initial_state = build_initial_state(args.pdb_id, args.pdb_file, work_dir, md_overrides, stage_overrides)

    agent_app = build_graph()
    result = agent_app.invoke(initial_state)

    print(f"✅ Final Status: {result['status']}")
    print(f"🔧 Final Config: {result['current_config']}")
    print(f"📜 History steps recorded: {len(result.get('history', []))}")
    print(f"📄 Reproduction script: {os.path.join(work_dir, 'reproduce.sh')}")


if __name__ == "__main__":
    main()
