# src/gromacs_agent/knowledge/db.py
import sqlite3
from typing import List, Dict, Optional, Tuple
import json
import re


class KnowledgeBase:
    """
    ハイブリッド型ナレッジベース。

    1. seed_initial_knowledge() で計算化学の既知ドメイン知識を初期投入 (Cold Start対策)
    2. record_result() で実行結果 (成功/失敗) を蓄積し success_count/fail_count を更新 (自己学習)

    success_count / fail_count は MCTS の UCB1 選択における事前分布 (prior) として
    mcts.search.MCTSStageSearch から利用される。これにより「過去に成功率が高かった
    修正」を優先的に探索でき、実行コストの高いGROMACS実行の回数を抑えられる。
    """

    def __init__(self, db_path: str = "knowledge.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY,
                    error_pattern TEXT,
                    cause TEXT,
                    solution TEXT,
                    stage TEXT DEFAULT '',
                    param_delta TEXT DEFAULT '{}',
                    success_count INTEGER DEFAULT 0,
                    fail_count INTEGER DEFAULT 0,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS errors_fts
                USING fts5(error_pattern, cause, solution)
            """)
            # 旧スキーマからのマイグレーション対策
            cols = {row[1] for row in conn.execute("PRAGMA table_info(errors)")}
            for col, ddl in [
                ("stage", "ALTER TABLE errors ADD COLUMN stage TEXT DEFAULT ''"),
                ("param_delta", "ALTER TABLE errors ADD COLUMN param_delta TEXT DEFAULT '{}'"),
                ("success_count", "ALTER TABLE errors ADD COLUMN success_count INTEGER DEFAULT 0"),
                ("fail_count", "ALTER TABLE errors ADD COLUMN fail_count INTEGER DEFAULT 0"),
            ]:
                if col not in cols:
                    conn.execute(ddl)
            self.seed_initial_knowledge()

    def seed_initial_knowledge(self):
        # 計算化学のドメイン知識をシード (stage / param_delta 付きで、そのままMCTSの
        # 候補パラメータとして使えるようにする)
        knowledge = [
            {
                "pattern": "LINCS WARNING",
                "cause": "Bond constraints failed. Time step (dt) too large or bad initial contacts.",
                "solution": "Decrease dt to 0.001 or increase EM steps.",
                "stage": "nvt",
                "param_delta": {"dt": 0.001},
            },
            {
                "pattern": "blowing up",
                "cause": "System instability due to steric clashes.",
                "solution": "Perform more aggressive Energy Minimization or use soft-core potentials.",
                "stage": "em",
                "param_delta": {"emtol": 500.0, "nsteps": 100000},
            },
            {
                "pattern": "Segmentation fault",
                "cause": "Memory issue or bug in GROMACS/MPI.",
                "solution": "Reduce system size or check hardware.",
                "stage": "",
                "param_delta": {},
            },
            {
                "pattern": "Energy minimization has stopped",
                "cause": "EM did not reach the target emtol within nsteps.",
                "solution": "Increase nsteps or relax emtol.",
                "stage": "em",
                "param_delta": {"nsteps": 200000, "emtol": 1000.0},
            },
            {
                "pattern": "pressure scaling",
                "cause": "Barostat/thermostat coupling too aggressive for the current state.",
                "solution": "Increase tau_t/tau_p or shorten dt.",
                "stage": "nvt",
                "param_delta": {"dt": 0.0005},
            },
        ]
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute("SELECT COUNT(*) FROM errors").fetchone()[0]
        if existing == 0:
            for k in knowledge:
                self.add_knowledge(
                    k["pattern"], k["cause"], k["solution"],
                    stage=k.get("stage", ""), param_delta=k.get("param_delta", {}),
                )

    def add_knowledge(self, pattern: str, cause: str, solution: str,
                       stage: str = "", param_delta: Optional[Dict] = None) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO errors (error_pattern, cause, solution, stage, param_delta) "
                "VALUES (?, ?, ?, ?, ?)",
                (pattern, cause, solution, stage, json.dumps(param_delta or {})),
            )
            rowid = cur.lastrowid
            conn.execute(
                "INSERT INTO errors_fts (rowid, error_pattern, cause, solution) VALUES (?, ?, ?, ?)",
                (rowid, pattern, cause, solution),
            )
            return rowid

    def search(self, error_log: str, stage: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """FTS5によるエラーログ検索。stageを指定すると該当ステージ向けの候補に絞り込む。"""
        query = self._sanitize_fts_query(error_log)
        if not query:
            return []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            sql = """
                SELECT e.id, e.error_pattern, e.cause, e.solution, e.stage, e.param_delta,
                       e.success_count, e.fail_count
                FROM errors_fts f
                JOIN errors e ON e.id = f.rowid
                WHERE errors_fts MATCH ?
            """
            params: list = [query]
            if stage:
                sql += " AND (e.stage = ? OR e.stage = '')"
                params.append(stage)
            sql += " LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            results = []
            for r in rows:
                visits = r["success_count"] + r["fail_count"]
                success_rate = (r["success_count"] / visits) if visits > 0 else 0.5
                results.append({
                    "id": r["id"],
                    "pattern": r["error_pattern"],
                    "cause": r["cause"],
                    "solution": r["solution"],
                    "stage": r["stage"],
                    "param_delta": json.loads(r["param_delta"] or "{}"),
                    "visits": visits,
                    "success_rate": success_rate,
                })
            return results

    def record_result(self, knowledge_id: Optional[int], success: bool):
        """MCTSのバックプロパゲーション結果をナレッジベースへ反映する (自己学習)。"""
        if knowledge_id is None:
            return
        col = "success_count" if success else "fail_count"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"UPDATE errors SET {col} = {col} + 1 WHERE id = ?", (knowledge_id,))

    def record_new_result(self, pattern: str, cause: str, solution: str,
                           stage: str, param_delta: Dict, success: bool) -> int:
        """未知のエラーパターンに対して自己発見した修正を新規レコードとして追記する。"""
        return self.add_knowledge(pattern, cause, solution, stage=stage, param_delta=param_delta) \
            if False else self._insert_with_outcome(pattern, cause, solution, stage, param_delta, success)

    def _insert_with_outcome(self, pattern, cause, solution, stage, param_delta, success) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO errors (error_pattern, cause, solution, stage, param_delta, "
                "success_count, fail_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (pattern, cause, solution, stage, json.dumps(param_delta),
                 1 if success else 0, 0 if success else 1),
            )
            rowid = cur.lastrowid
            conn.execute(
                "INSERT INTO errors_fts (rowid, error_pattern, cause, solution) VALUES (?, ?, ?, ?)",
                (rowid, pattern, cause, solution),
            )
            return rowid

    def get_prior(self, knowledge_id: Optional[int]) -> Tuple[int, int]:
        """(visits, success_count) を返す。MCTSノード初期化時の事前分布として使用する。"""
        if knowledge_id is None:
            return (0, 0)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT success_count, fail_count FROM errors WHERE id = ?", (knowledge_id,)
            ).fetchone()
            if not row:
                return (0, 0)
            success_count, fail_count = row
            return (success_count + fail_count, success_count)

    @staticmethod
    def _sanitize_fts_query(text: str) -> str:
        """FTS5のクエリ構文エラーを避けるため記号を除去し、空白区切りのORクエリに変換する。"""
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text or "")
        stop = {"the", "and", "for", "was", "with", "step", "gromacs"}
        tokens = [t for t in tokens[:20] if t.lower() not in stop]
        if not tokens:
            return ""
        return " OR ".join(sorted(set(tokens)))
