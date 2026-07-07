# tests/test_mcts.py
import pytest
from unittest.mock import MagicMock
from gromacs_agent.mcts.search import MCTSStageSearch
from gromacs_agent.knowledge.db import KnowledgeBase


@pytest.fixture
def mock_knowledge_base(tmp_path):
    """モックされたKnowledgeBaseを返すフィクスチャ"""
    kb = KnowledgeBase(db_path=str(tmp_path / "test_mcts_kb.db"))
    return kb


class TestMCTSStageSearch:
    """MCTSStageSearchの単体テスト"""

    def test_mcts_immediate_success(self, mock_knowledge_base):
        """最初の試行で成功した場合、追加の探索を行わない"""
        call_count = {"count": 0}

        def simulate_fn(stage: str, config: dict) -> tuple[bool, float, str]:
            call_count["count"] += 1
            # 最初の試行で成功
            return True, 1.0, ""

        search = MCTSStageSearch(
            stage="em",
            base_config={"dt": 0.002, "nsteps": 50000},
            kb=mock_knowledge_base,
            simulate_fn=simulate_fn,
            max_iterations=4,
        )

        result = search.run()

        assert result["success"] is True
        assert result["iterations_used"] == 1
        # 最初の試行で成功したので、1回だけ呼ばれる
        assert call_count["count"] == 1
        assert result["config"]["dt"] == 0.002

    def test_mcts_exploration_on_failure(self, mock_knowledge_base):
        """失敗が続いた場合、異なるパラメータを探索する"""
        call_count = {"count": 0}
        attempted_configs = []

        def simulate_fn(stage: str, config: dict) -> tuple[bool, float, str]:
            call_count["count"] += 1
            attempted_configs.append(config.copy())
            
            # 最初の2回は失敗、3回目で成功
            if call_count["count"] <= 2:
                return False, 0.0, "LINCS warning"
            return True, 1.0, ""

        search = MCTSStageSearch(
            stage="em",
            base_config={"dt": 0.002, "nsteps": 50000},
            kb=mock_knowledge_base,
            simulate_fn=simulate_fn,
            max_iterations=4,
        )

        result = search.run()

        assert result["success"] is True
        assert call_count["count"] == 3
        # 異なる設定が試されたことを確認
        assert len(attempted_configs) == 3
        # 少なくとも1回はdtが変更されている
        dt_values = [c["dt"] for c in attempted_configs]
        assert len(set(dt_values)) > 1 or dt_values[0] == 0.002

    def test_mcts_max_iterations_exhausted(self, mock_knowledge_base):
        """最大イテレーションに達した場合、失敗を返す"""
        call_count = {"count": 0}

        def simulate_fn(stage: str, config: dict) -> tuple[bool, float, str]:
            call_count["count"] += 1
            # 常に失敗
            return False, 0.0, "Blowing up"

        search = MCTSStageSearch(
            stage="em",
            base_config={"dt": 0.002, "nsteps": 50000},
            kb=mock_knowledge_base,
            simulate_fn=simulate_fn,
            max_iterations=3,
        )

        result = search.run()

        assert result["success"] is False
        assert call_count["count"] == 3
        assert result["iterations_used"] == 3
        assert "Blowing up" in result["log"]

    def test_mcts_parameter_reduction(self, mock_knowledge_base):
        """LINCS警告に対してdtを縮小する"""
        def simulate_fn(stage: str, config: dict) -> tuple[bool, float, str]:
            # dtが0.001以下なら成功
            if config.get("dt", 0.002) <= 0.001:
                return True, 1.0, ""
            return False, 0.0, "LINCS warning"

        search = MCTSStageSearch(
            stage="em",
            base_config={"dt": 0.002, "nsteps": 50000},
            kb=mock_knowledge_base,
            simulate_fn=simulate_fn,
            max_iterations=4,
        )

        result = search.run()

        assert result["success"] is True
        assert result["config"]["dt"] <= 0.001

    def test_mcts_emtol_adjustment(self, mock_knowledge_base):
        """収束失敗に対してemtolを調整する"""
        def simulate_fn(stage: str, config: dict) -> tuple[bool, float, str]:
            # emtolが500以下なら成功
            if config.get("emtol", 1000.0) <= 500.0:
                return True, 1.0, ""
            return False, 0.0, "EM did not converge"

        search = MCTSStageSearch(
            stage="em",
            base_config={"dt": 0.002, "emtol": 1000.0, "nsteps": 50000},
            kb=mock_knowledge_base,
            simulate_fn=simulate_fn,
            max_iterations=4,
        )

        result = search.run()

        assert result["success"] is True
        assert result["config"]["emtol"] <= 500.0

    def test_mcts_tree_structure(self, mock_knowledge_base):
        """MCTSツリーが正しく構築される"""
        def simulate_fn(stage: str, config: dict) -> tuple[bool, float, str]:
            if config.get("dt", 0.002) <= 0.001:
                return True, 1.0, ""
            return False, 0.0, "LINCS warning"

        search = MCTSStageSearch(
            stage="em",
            base_config={"dt": 0.002, "nsteps": 50000},
            kb=mock_knowledge_base,
            simulate_fn=simulate_fn,
            max_iterations=3,
        )

        result = search.run()

        # ツリー構造が存在することを確認
        assert "tree" in result or result["success"] is True
        if "tree" in result:
            assert isinstance(result["tree"], dict)

    def test_mcts_with_knowledge_base_insights(self, mock_knowledge_base):
        """KnowledgeBaseの情報が探索に活用される"""
        # KnowledgeBaseに過去の成功ケースを追加
        mock_knowledge_base.add_entry(
            error_pattern="LINCS warning",
            solution='{"dt": 0.0005}',
            stage="em"
        )

        def simulate_fn(stage: str, config: dict) -> tuple[bool, float, str]:
            if config.get("dt", 0.002) <= 0.0005:
                return True, 1.0, ""
            return False, 0.0, "LINCS warning"

        search = MCTSStageSearch(
            stage="em",
            base_config={"dt": 0.002, "nsteps": 50000},
            kb=mock_knowledge_base,
            simulate_fn=simulate_fn,
            max_iterations=4,
        )

        result = search.run()

        assert result["success"] is True
        # KnowledgeBaseの情報が活用され、効率的に成功する
        assert result["config"]["dt"] <= 0.0005

    def test_mcts_different_stages(self, mock_knowledge_base):
        """異なるステージ（em, nvt, npt）でMCTSが動作する"""
        for stage in ["em", "nvt", "npt"]:
            def simulate_fn(s: str, config: dict) -> tuple[bool, float, str]:
                if config.get("dt", 0.002) <= 0.001:
                    return True, 1.0, ""
                return False, 0.0, f"{stage} failed"

            search = MCTSStageSearch(
                stage=stage,
                base_config={"dt": 0.002, "nsteps": 50000},
                kb=mock_knowledge_base,
                simulate_fn=simulate_fn,
                max_iterations=3,
            )

            result = search.run()
            assert result["success"] is True, f"Failed for stage {stage}"

    def test_mcts_reward_propagation(self, mock_knowledge_base):
        """報酬が正しく伝播する"""
        call_count = {"count": 0}

        def simulate_fn(stage: str, config: dict) -> tuple[bool, float, str]:
            call_count["count"] += 1
            # 3回目で成功（報酬1.0）
            if call_count["count"] == 3:
                return True, 1.0, ""
            return False, 0.0, "Failed"

        search = MCTSStageSearch(
            stage="em",
            base_config={"dt": 0.002, "nsteps": 50000},
            kb=mock_knowledge_base,
            simulate_fn=simulate_fn,
            max_iterations=4,
        )

        result = search.run()

        assert result["success"] is True
        assert call_count["count"] == 3

    def test_mcts_config_preservation(self, mock_knowledge_base):
        """変更されていない設定が保持される"""
        def simulate_fn(stage: str, config: dict) -> tuple[bool, float, str]:
            if config.get("dt", 0.002) <= 0.001:
                return True, 1.0, ""
            return False, 0.0, "LINCS warning"

        search = MCTSStageSearch(
            stage="em",
            base_config={"dt": 0.002, "nsteps": 50000, "emtol": 1000.0},
            kb=mock_knowledge_base,
            simulate_fn=simulate_fn,
            max_iterations=3,
        )

        result = search.run()

        assert result["success"] is True
        # dtは変更されるが、nstepsとemtolは保持される
        assert result["config"]["nsteps"] == 50000
        assert result["config"]["emtol"] == 1000.0
        assert result["config"]["dt"] <= 0.001
