# src/gromacs_agent/core/config.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List

class GromacsConfig(BaseModel):
    force_field: str = Field(default="amber99sb-ildn")
    water_model: str = Field(default="tip3p")

class SimulationStepConfig(BaseModel):
    nsteps: int
    dt: float = 0.002
    emtol: float = 1000.0

class MCTSConfig(BaseModel):
    """em/nvtなど『切りの良い単位』でのステージ内MCTS探索の設定。"""
    stages: List[str] = Field(
        default=["em", "nvt"],
        description="MCTSで複数パラメータ候補を探索するステージ名のリスト",
    )
    max_iterations: int = Field(
        default=4, description="1ステージあたりの最大実プレイアウト回数 (実際のgmx実行回数の上限)"
    )
    max_candidates: int = Field(
        default=3, description="1回の展開(Expansion)で生成する子ノード候補数の上限"
    )
    exploration_constant: float = Field(
        default=1.41, description="UCB1の探索定数 C (デフォルト sqrt(2))"
    )
    max_depth: int = Field(
        default=3, description="連鎖的な修正探索を許す最大深さ"
    )

class AgentConfig(BaseModel):
    max_attempts: int = Field(default=3, description="最大リトライ回数 (MCTS対象外ステージ)")
    llm_provider: str = Field(default="openai", description="openai or anthropic")
    model_name: str = Field(default="gpt-4o")
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
