# tests/test_agent.py
import pytest
from unittest.mock import patch, MagicMock
from gromacs_agent.core.graph import build_graph
from gromacs_agent.core.state import AgentState

@pytest.fixture
def mock_gmx():
    with patch("gromacs_agent.tools.gromacs_tools.subprocess.run") as mock_run:
        yield mock_run

def test_agent_replan_on_lincs_warning(mock_gmx):
    # モックの設定: 最初はLINCSエラーで失敗、2回目は成功
    def side_effect(*args, **kwargs):
        if mock_gmx.call_count == 1:
            # 失敗パターン
            res = MagicMock()
            res.returncode = 1
            res.stderr = "LINCS WARNING: step 0"
            return res
        else:
            # 成功パターン
            res = MagicMock()
            res.returncode = 0
            return res

    mock_gmx.side_effect = side_effect

    graph = build_graph()
    initial_state = {
        "system_name": "test",
        "max_attempts": 3,
        "current_config": {"dt": 0.002},
        "attempt_count": 0,
        "workflow": ["em"]
    }
    
    # 実行
    result = graph.invoke(initial_state)
    
    # 検証
    assert result["attempt_count"] > 0
    assert result["status"] == "SUCCESS"
    assert result["current_config"]["dt"] == 0.001 # Replannerが修正したはず
