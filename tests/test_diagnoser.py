# tests/test_diagnoser.py
import json
import pytest
from unittest.mock import patch, MagicMock
from gromacs_agent.nodes.diagnoser import diagnose_node
from gromacs_agent.knowledge.db import KnowledgeBase


@pytest.fixture
def mock_knowledge_base(tmp_path):
    """モックされたKnowledgeBaseを返すフィクスチャ"""
    kb = KnowledgeBase(db_path=str(tmp_path / "test_kb.db"))
    return kb


class TestDiagnoserWithLLM:
    """LLMを使用した診断のテスト"""

    @patch("gromacs_agent.nodes.diagnoser.ChatPromptTemplate")
    @patch("gromacs_agent.nodes.diagnoser.ChatOpenAI")
    def test_diagnoser_with_llm_success(self, mock_chat_openai, mock_prompt_template, mock_knowledge_base, monkeypatch):
        """LLMが正常に診断結果を返すケース"""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        monkeypatch.setattr(
            "gromacs_agent.nodes.diagnoser.KnowledgeBase",
            lambda: mock_knowledge_base
        )
        
        # LLMのモックレスポンスを設定
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "cause": "LINCS warning due to large time step",
            "fix_type": "PARAMETER_CHANGE",
            "parameters": {"dt": 0.001, "emtol": 500.0}
        })
        
        # チェーン（prompt | llm）のモック
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response
        
        # ChatPromptTemplate.from_template().invoke() が呼ばれるので、
        # from_template の戻り値をモックし、その __or__ メソッドでモックチェーンを返す
        mock_prompt = MagicMock()
        mock_prompt.__or__.return_value = mock_chain
        mock_prompt_template.from_template.return_value = mock_prompt
        
        mock_llm_instance = MagicMock()
        mock_chat_openai.return_value = mock_llm_instance
        
        state = {
            "status": "FAILED",
            "last_error": "LINCS warning: too many warnings",
            "current_config": {"dt": 0.002, "nsteps": 50000},
        }
        
        result = diagnose_node(state)
        
        assert result["status"] == "NEEDS_REPLAN"
        assert "diagnosis_context" in result
        assert result["diagnosis_context"]["cause"] == "LINCS warning due to large time step"
        assert result["diagnosis_context"]["fix_type"] == "PARAMETER_CHANGE"
        assert result["diagnosis_context"]["parameters"]["dt"] == 0.001
        assert result["diagnosis_context"]["parameters"]["emtol"] == 500.0
        
        mock_chat_openai.assert_called_once_with(model="gpt-4o", temperature=0)
        assert mock_chain.invoke.called

    @patch("gromacs_agent.nodes.diagnoser.ChatPromptTemplate")
    @patch("gromacs_agent.nodes.diagnoser.ChatOpenAI")
    def test_diagnoser_with_llm_parse_error(self, mock_chat_openai, mock_prompt_template, mock_knowledge_base, monkeypatch):
        """LLMが不正なJSONを返すケース（フォールバック処理が動作することを確認）"""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        monkeypatch.setattr(
            "gromacs_agent.nodes.diagnoser.KnowledgeBase",
            lambda: mock_knowledge_base
        )
        
        # 不正なJSONを返すモック
        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON {{{"
        
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response
        
        mock_prompt = MagicMock()
        mock_prompt.__or__.return_value = mock_chain
        mock_prompt_template.from_template.return_value = mock_prompt
        
        mock_llm_instance = MagicMock()
        mock_chat_openai.return_value = mock_llm_instance
        
        state = {
            "status": "FAILED",
            "last_error": "Segmentation fault in mdrun",
        }
        
        result = diagnose_node(state)
        
        assert result["status"] == "NEEDS_REPLAN"
        assert "diagnosis_context" in result
        assert "Parse error" in result["diagnosis_context"]["cause"]
        assert result["diagnosis_context"]["fix_type"] == "PARAMETER_CHANGE"
        assert "dt" in result["diagnosis_context"]["parameters"]

    @patch("gromacs_agent.nodes.diagnoser.ChatPromptTemplate")
    @patch("gromacs_agent.nodes.diagnoser.ChatOpenAI")
    def test_diagnoser_with_api_error(self, mock_chat_openai, mock_prompt_template, mock_knowledge_base, monkeypatch):
        """APIエラーが発生してフォールバックするケース"""
        from openai import AuthenticationError
        
        monkeypatch.setenv("OPENAI_API_KEY", "invalid_key")
        monkeypatch.setattr(
            "gromacs_agent.nodes.diagnoser.KnowledgeBase",
            lambda: mock_knowledge_base
        )
        
        # APIエラーを発生させるモック
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = AuthenticationError(
            message="Invalid API key",
            response=MagicMock(),
            body={"error": {"message": "Invalid API key"}}
        )
        
        mock_prompt = MagicMock()
        mock_prompt.__or__.return_value = mock_chain
        mock_prompt_template.from_template.return_value = mock_prompt
        
        mock_llm_instance = MagicMock()
        mock_chat_openai.return_value = mock_llm_instance
        
        state = {
            "status": "FAILED",
            "last_error": "Blowing up: instability detected",
        }
        
        result = diagnose_node(state)
        
        assert result["status"] == "NEEDS_REPLAN"
        assert "diagnosis_context" in result
        assert "OpenAI error" in result["diagnosis_context"]["cause"]
        assert result["diagnosis_context"]["fix_type"] == "PARAMETER_CHANGE"
        assert "emtol" in result["diagnosis_context"]["parameters"]
        assert "nsteps" in result["diagnosis_context"]["parameters"]

    @patch("gromacs_agent.nodes.diagnoser.ChatPromptTemplate")
    @patch("gromacs_agent.nodes.diagnoser.ChatOpenAI")
    def test_diagnoser_with_lincs_warning(self, mock_chat_openai, mock_prompt_template, mock_knowledge_base, monkeypatch):
        """LINCS警告に対する診断が正しく行われるケース"""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        monkeypatch.setattr(
            "gromacs_agent.nodes.diagnoser.KnowledgeBase",
            lambda: mock_knowledge_base
        )
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "cause": "LINCS algorithm failed due to large time step",
            "fix_type": "PARAMETER_CHANGE",
            "parameters": {"dt": 0.0005}
        })
        
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response
        
        mock_prompt = MagicMock()
        mock_prompt.__or__.return_value = mock_chain
        mock_prompt_template.from_template.return_value = mock_prompt
        
        mock_llm_instance = MagicMock()
        mock_chat_openai.return_value = mock_llm_instance
        
        state = {
            "status": "FAILED",
            "last_error": "LINCS warning: too many warnings, 100% of the bonds changed length",
        }
        
        result = diagnose_node(state)
        
        assert result["status"] == "NEEDS_REPLAN"
        assert result["diagnosis_context"]["parameters"]["dt"] == 0.0005

    @patch("gromacs_agent.nodes.diagnoser.ChatPromptTemplate")
    @patch("gromacs_agent.nodes.diagnoser.ChatOpenAI")
    def test_diagnoser_with_workflow_change_suggestion(self, mock_chat_openai, mock_prompt_template, mock_knowledge_base, monkeypatch):
        """LLMがワークフロー変更を提案するケース"""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        monkeypatch.setattr(
            "gromacs_agent.nodes.diagnoser.KnowledgeBase",
            lambda: mock_knowledge_base
        )
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "cause": "Missing position restraints in initial setup",
            "fix_type": "WORKFLOW_CHANGE",
            "parameters": {"add_position_restraints": True}
        })
        
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response
        
        mock_prompt = MagicMock()
        mock_prompt.__or__.return_value = mock_chain
        mock_prompt_template.from_template.return_value = mock_prompt
        
        mock_llm_instance = MagicMock()
        mock_chat_openai.return_value = mock_llm_instance
        
        state = {
            "status": "FAILED",
            "last_error": "System blew up during NVT equilibration",
        }
        
        result = diagnose_node(state)
        
        assert result["status"] == "NEEDS_REPLAN"
        assert result["diagnosis_context"]["fix_type"] == "WORKFLOW_CHANGE"
        assert result["diagnosis_context"]["parameters"]["add_position_restraints"] is True

    def test_diagnoser_without_api_key(self, mock_knowledge_base, monkeypatch):
        """APIキーが設定されていない場合のフォールバック処理"""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(
            "gromacs_agent.nodes.diagnoser.KnowledgeBase",
            lambda: mock_knowledge_base
        )
        
        state = {
            "status": "FAILED",
            "last_error": "LINCS warning detected",
        }
        
        result = diagnose_node(state)
        
        assert result["status"] == "NEEDS_REPLAN"
        assert "diagnosis_context" in result
        assert "no API key" in result["diagnosis_context"]["cause"]
        assert result["diagnosis_context"]["fix_type"] == "PARAMETER_CHANGE"
        assert result["diagnosis_context"]["parameters"]["dt"] == 0.001

    def test_diagnoser_with_success_status(self, mock_knowledge_base, monkeypatch):
        """ステータスがSUCCESSの場合は診断を行わない"""
        monkeypatch.setattr(
            "gromacs_agent.nodes.diagnoser.KnowledgeBase",
            lambda: mock_knowledge_base
        )
        
        state = {
            "status": "SUCCESS",
            "last_error": None,
        }
        
        result = diagnose_node(state)
        
        assert result["status"] == "SUCCESS"
        assert "diagnosis_context" not in result
