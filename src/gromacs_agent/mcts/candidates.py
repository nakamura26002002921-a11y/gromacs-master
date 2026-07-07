# src/gromacs_agent/mcts/candidates.py
"""
MCTSの「展開 (Expansion)」フェーズで使う候補パラメータ生成。

方針:
  1. まずKnowledgeBase.search()でヒットしたレコード (KB由来の実績ある修正) を候補にする。
  2. ヒットが少ない/ゼロの場合は、ステージ固有のヒューリスティックで補完する
     (安全な探索空間に留めるためのフォールバック。LLM呼び出し無しで動作する)。
  3. LLMによる診断 (diagnoser) が利用可能な場合は追加の候補として合流させる
     (任意・省略可能。ネットワーク/APIキーが無い環境でもMCTS自体は動作させたいため)。
"""
from typing import Dict, List, Optional


# ステージごとの安全なヒューリスティック候補 (KBに何もヒットしなかった場合のフォールバック)
_HEURISTIC_LIBRARY = {
    "em": [
        {"emtol": 500.0},
        {"nsteps": 100000},
        {"emtol": 500.0, "nsteps": 100000},
    ],
    "nvt": [
        {"dt": 0.001},
        {"dt": 0.0005},
        {"tau_t": 0.2},
    ],
    "npt": [
        {"dt": 0.001},
        {"tau_p": 2.0},
    ],
    "md": [
        {"dt": 0.001},
    ],
}


def generate_candidates(
    stage: str,
    base_config: Dict,
    error_log: str,
    kb,
    max_candidates: int = 4,
) -> List[Dict]:
    """
    展開候補のリストを返す。各要素は
      {"config": {...マージ済み設定...}, "knowledge_id": Optional[int], "source": str}
    """
    candidates: List[Dict] = []
    seen_signatures = set()

    def _add(delta: Dict, knowledge_id: Optional[int], source: str):
        if not delta:
            return
        merged = {**base_config, **delta}
        sig = tuple(sorted(merged.items()))
        if sig in seen_signatures:
            return
        seen_signatures.add(sig)
        candidates.append({"config": merged, "knowledge_id": knowledge_id, "source": source})

    # 1. KnowledgeBase (実績ベース、success_rateの高い順)
    kb_hits = kb.search(error_log, stage=stage, limit=max_candidates) if error_log else []
    for hit in sorted(kb_hits, key=lambda h: h.get("success_rate", 0.5), reverse=True):
        _add(hit.get("param_delta", {}), hit.get("id"), "kb")

    # 2. ヒューリスティックで補完 (KB候補が不足している場合のみ)
    if len(candidates) < max_candidates:
        for delta in _HEURISTIC_LIBRARY.get(stage, []):
            if len(candidates) >= max_candidates:
                break
            _add(delta, None, "heuristic")

    return candidates[:max_candidates]
