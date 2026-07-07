# src/gromacs_agent/knowledge/db.py
import sqlite3
from typing import List, Dict
import json

class KnowledgeBase:
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
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS errors_fts 
                USING fts5(error_pattern, cause, solution)
            """)
            self.seed_initial_knowledge()

    def seed_initial_knowledge(self):
        # 計算化学のドメイン知識をシード
        knowledge = [
            {
                "pattern": "LINCS WARNING",
                "cause": "Bond constraints failed. Time step (dt) too large or bad initial contacts.",
                "solution": "Decrease dt to 0.001 or increase EM steps."
            },
            {
                "pattern": "blowing up",
                "cause": "System instability due to steric clashes.",
                "solution": "Perform more aggressive Energy Minimization or use soft-core potentials."
            },
            {
                "pattern": "Segmentation fault",
                "cause": "Memory issue or bug in GROMACS/MPI.",
                "solution": "Reduce system size or check hardware."
            }
        ]
        for k in knowledge:
            self.add_knowledge(k["pattern"], k["cause"], k["solution"])

    def add_knowledge(self, pattern: str, cause: str, solution: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO errors (error_pattern, cause, solution) VALUES (?, ?, ?)",
                         (pattern, cause, solution))
            conn.execute("INSERT INTO errors_fts (rowid, error_pattern, cause, solution) VALUES (last_insert_rowid(), ?, ?, ?)",
                         (pattern, cause, solution))

    def search(self, error_log: str, limit: int = 3) -> List[Dict[str, str]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT error_pattern, cause, solution FROM errors_fts WHERE errors_fts MATCH ? LIMIT ?",
                (error_log, limit)
            )
            return [{"pattern": r[0], "cause": r[1], "solution": r[2]} for r in cursor.fetchall()]
