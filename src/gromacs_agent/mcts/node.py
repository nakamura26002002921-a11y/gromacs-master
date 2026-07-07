# src/gromacs_agent/mcts/node.py
import math
from typing import Dict, List, Optional


class MCTSNode:
    """
    MCTS探索木の1ノード = 「あるステージ (em/nvt等) における1つのパラメータ候補」。

    - config: そのステージで実際にgromacsへ渡すパラメータ (dt, nsteps, emtol等)
    - knowledge_id: 由来となったKnowledgeBaseレコードのid (自己学習の記録先)
    - visits / wins: このノードの実プレイアウト (=実際のgmx実行) 回数と成功回数
    - prior_visits / prior_wins: KnowledgeBaseから得た事前分布 (仮想的な事前試行回数)
    """

    def __init__(
        self,
        stage: str,
        config: Dict,
        parent: Optional["MCTSNode"] = None,
        knowledge_id: Optional[int] = None,
        source: str = "root",
    ):
        self.stage = stage
        self.config = config
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.knowledge_id = knowledge_id
        self.source = source  # "root" | "kb" | "heuristic" | "llm"

        self.visits = 0
        self.wins = 0.0  # 部分報酬を許すためfloat (0.0〜1.0の合計)

        # KnowledgeBaseの成功率をUCB1の事前分布として使う (仮想試行)
        self.prior_visits = 0
        self.prior_wins = 0.0

        self.result_log: Optional[str] = None
        self.is_terminal_success = False

    def set_prior(self, prior_visits: int, prior_wins: float):
        self.prior_visits = prior_visits
        self.prior_wins = prior_wins

    @property
    def effective_visits(self) -> int:
        return self.visits + self.prior_visits

    @property
    def effective_wins(self) -> float:
        return self.wins + self.prior_wins

    @property
    def value(self) -> float:
        """平均報酬 (事前分布込みの成功率)。"""
        ev = self.effective_visits
        return self.effective_wins / ev if ev > 0 else 0.5

    def ucb1(self, exploration_constant: float, parent_visits: int) -> float:
        """
        UCB1スコア。未実行 (effective_visits==0) のノードは最優先で探索する。
        prior_visitsがある場合はKB由来の「既にある程度試された」ノードとして
        自然に探索と活用のバランスが取れる。
        """
        if self.effective_visits == 0:
            return float("inf")
        exploitation = self.value
        exploration = exploration_constant * math.sqrt(
            math.log(max(parent_visits, 1)) / self.effective_visits
        )
        return exploitation + exploration

    def best_child_by_ucb(self, exploration_constant: float) -> "MCTSNode":
        parent_visits = max(self.effective_visits, 1)
        return max(self.children, key=lambda c: c.ucb1(exploration_constant, parent_visits))

    def most_robust_child(self) -> Optional["MCTSNode"]:
        """探索終了時、最も試行回数が多く成功率が高いノードを最終選択する (Robust Child)。"""
        if not self.children:
            return None
        return max(self.children, key=lambda c: (c.wins > 0, c.effective_visits, c.value))

    def backpropagate(self, reward: float):
        node: Optional["MCTSNode"] = self
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent

    def to_dict(self) -> Dict:
        return {
            "stage": self.stage,
            "config": self.config,
            "source": self.source,
            "visits": self.visits,
            "wins": self.wins,
            "prior_visits": self.prior_visits,
            "prior_wins": self.prior_wins,
            "value": round(self.value, 4),
            "children": [c.to_dict() for c in self.children],
        }
