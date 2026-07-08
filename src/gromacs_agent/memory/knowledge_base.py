# src/gromacs_agent/memory/knowledge_base.py
import json
import os
import re
from typing import List, Dict, Optional

class KnowledgeBase:
    def __init__(self, db_path: str = "gromacs_kb.json"):
        self.db_path = db_path
        self.data = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                return json.load(f)
        return {"successes": []}

    def _save(self):
        with open(self.db_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def _extract_error_signature(self, stderr: str) -> str:
        """エラーメッセージから特徴的なキーワード（Fatal error以降など）を抽出"""
        match = re.search(r"Fatal error:\n(.*)", stderr, re.DOTALL)
        if match:
            return match.group(1).strip()[:200] # 長すぎる場合は切り詰め
        return stderr.strip()[-200:]

    def search_similar_fixes(self, stage: str, stderr: str) -> List[Dict]:
        """過去の成功事例から類似の修正方法を検索"""
        sig = self._extract_error_signature(stderr)
        candidates = []
        
        for entry in self.data["successes"]:
            if entry["stage"] == stage:
                # 簡易スコアリング（共通する単語の数をカウント）
                score = sum(1 for word in sig.split() if word in entry["error_sig"])
                if score > 3: # 閾値
                    candidates.append({**entry, "score": score})
                    
        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:3]

    def add_success(self, stage: str, stderr: str, original_args: list, fixed_args: list, reason: str):
        """成功した修正をKBに追加"""
        entry = {
            "stage": stage,
            "error_sig": self._extract_error_signature(stderr),
            "original_args": original_args,
            "fixed_args": fixed_args,
            "reason": reason
        }
        self.data["successes"].append(entry)
        self._save()
