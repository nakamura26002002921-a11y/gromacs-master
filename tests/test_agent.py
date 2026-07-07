# tests/test_agent.py
import re
import pytest
from unittest.mock import patch, MagicMock

from gromacs_agent.core.graph import build_graph
from gromacs_agent.knowledge.db import KnowledgeBase


def _deffnm(full_cmd):
    if "-deffnm" in full_cmd:
        return full_cmd[full_cmd.index("-deffnm") + 1]
    return None


def _dt(full_cmd):
    if "-dt" in full_cmd:
        return float(full_cmd[full_cmd.index("-dt") + 1])
    return None


def _mock_subprocess(fail_first_attempt_stages=("em", "nvt"), always_fail_stages=(), fail_message="LINCS WARNING: step 0"):
    """
    grompp呼び出しは常に成功させる。mdrun呼び出しは:
      - always_fail_stages に含まれるステージは常に失敗させる (budget枯渇シナリオ用)
      - fail_first_attempt_stages に含まれるステージは、そのステージの1回目の実行
        (=ステージ初期パラメータでの実行) のときのみ失敗させ、MCTSが展開した
        いずれかの候補 (2回目以降の試行) では成功する (探索による回復シナリオ用)。
        どのパラメータが「正解」かは問わない = MCTSの配線そのものを検証する目的。
    """
    attempt_counts = {}

    def side_effect(full_cmd, **kwargs):
        res = MagicMock()
        if full_cmd[1] == "grompp":
            res.returncode, res.stdout, res.stderr = 0, "grompp OK", ""
            return res

        stage = _deffnm(full_cmd)
        attempt_counts[stage] = attempt_counts.get(stage, 0) + 1

        if stage in always_fail_stages:
            res.returncode, res.stdout, res.stderr = 1, "", fail_message
        elif stage in fail_first_attempt_stages and attempt_counts[stage] == 1:
            res.returncode, res.stdout, res.stderr = 1, "", fail_message
        else:
            res.returncode, res.stdout, res.stderr = 0, "Finished mdrun successfully", ""
        return res

    return side_effect


@patch("gromacs_agent.tools.gromacs_tools.subprocess.run")
def test_mcts_stage_recovers_from_lincs_warning(mock_run, tmp_path, monkeypatch):
    """em/nvtで最初の設定(dt=0.002)が失敗しても、MCTSが展開した候補(dt縮小)で
    回復し、ワークフロー全体が最後まで完走することを確認する。"""
    monkeypatch.setattr(
        "gromacs_agent.nodes.mcts_stage.KnowledgeBase",
        lambda: KnowledgeBase(db_path=str(tmp_path / "kb.db")),
    )
    mock_run.side_effect = _mock_subprocess(fail_first_attempt_stages=("em", "nvt"))

    graph = build_graph()
    initial_state = {
        "system_name": "test",
        "current_config": {"dt": 0.002, "nsteps": 50000},
        "history": [],
        "mcts_stages": ["em", "nvt"],
        "mcts_max_iterations": 4,
    }

    result = graph.invoke(initial_state)

    assert result["status"] == "ALL_DONE"
    assert result["step_index"] == len(result["workflow"])

    mcts_history = [h for h in result["history"] if h.get("type") == "mcts"]
    stages_recorded = {h["stage"] for h in mcts_history}
    assert stages_recorded == {"em", "nvt"}
    assert all(h["success"] for h in mcts_history)
    # 各ステージともMCTSが最低1回は展開(候補を試す)を行っているはず
    assert all(h["iterations_used"] >= 2 for h in mcts_history)


@patch("gromacs_agent.tools.gromacs_tools.subprocess.run")
def test_mcts_stage_fails_after_exhausting_budget(mock_run, tmp_path, monkeypatch):
    """emが常に失敗し続ける場合、MCTSがbudgetを使い切ってSTAGE_FAILEDになることを確認する。"""
    monkeypatch.setattr(
        "gromacs_agent.nodes.mcts_stage.KnowledgeBase",
        lambda: KnowledgeBase(db_path=str(tmp_path / "kb2.db")),
    )
    mock_run.side_effect = _mock_subprocess(always_fail_stages=("em",), fail_message="blowing up")

    graph = build_graph()
    initial_state = {
        "system_name": "test",
        "current_config": {"dt": 0.002, "nsteps": 50000},
        "history": [],
        "mcts_stages": ["em", "nvt"],
        "mcts_max_iterations": 3,
    }

    result = graph.invoke(initial_state)

    assert result["status"] == "STAGE_FAILED"
    mcts_history = [h for h in result["history"] if h.get("type") == "mcts"]
    assert mcts_history[-1]["stage"] == "em"
    assert mcts_history[-1]["success"] is False


@patch("gromacs_agent.tools.gromacs_tools.subprocess.run")
def test_non_mcts_stage_uses_simple_retry_loop(mock_run, tmp_path, monkeypatch):
    """pdb2gmx等MCTS対象外ステージは、従来通りdiagnoser/replannerの1本道リトライで回復する。
    リトライ自体が機能することの確認が目的なので、修正の正しさに依らず
    2回目の呼び出しで成功するモックにしている。"""
    monkeypatch.setattr(
        "gromacs_agent.nodes.diagnoser.KnowledgeBase",
        lambda: KnowledgeBase(db_path=str(tmp_path / "kb3.db")),
    )
    calls = {"mdrun": 0}

    def side_effect(full_cmd, **kwargs):
        res = MagicMock()
        if full_cmd[1] == "grompp":
            res.returncode, res.stdout, res.stderr = 0, "grompp OK", ""
            return res
        calls["mdrun"] += 1
        if _deffnm(full_cmd) == "pdb2gmx" and calls["mdrun"] == 1:
            res.returncode, res.stdout, res.stderr = 1, "", "Segmentation fault"
        else:
            res.returncode, res.stdout, res.stderr = 0, "Finished mdrun successfully", ""
        return res

    mock_run.side_effect = side_effect

    graph = build_graph()
    initial_state = {
        "system_name": "test",
        "current_config": {"dt": 0.002, "nsteps": 50000},
        "history": [],
        "mcts_stages": [],  # 全ステージをMCTS対象外にする
        "max_attempts": 3,
    }

    result = graph.invoke(initial_state)

    assert result["status"] == "ALL_DONE"
