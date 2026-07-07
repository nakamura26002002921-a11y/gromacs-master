# src/gromacs_agent/mcts/search.py
"""
ステージ単位 (em, nvt, npt, md...) のMCTS探索。

GROMACSの1回の実行 (grompp+mdrun) は数分〜数時間かかるため、通常のMCTSのような
「軽量なランダムロールアウトを大量に行う」設計は取れない。そのため:

  - 深さは基本的に浅く抑える (max_depth, デフォルト2〜3)
  - 1回のシミュレーション = 実際のgmx実行 (本物のプレイアウト)
  - KnowledgeBaseのsuccess_rateを事前分布 (prior) として使い、
    「筋の良い候補」から優先的に実行することで実行回数そのものを削減する
  - 子ノードが1つでも成功したら、そのステージは即座に完了とみなし探索を打ち切る
    (=EM/NVTなど「切りの良い単位」でのチェックポイント運用)
"""
from typing import Callable, Dict, List, Optional, Tuple

from gromacs_agent.mcts.node import MCTSNode
from gromacs_agent.mcts.candidates import generate_candidates

# シミュレーション結果を表す型: (success, reward[0.0-1.0], log_text)
SimulateFn = Callable[[str, Dict], Tuple[bool, float, str]]


class MCTSStageSearch:
    def __init__(
        self,
        stage: str,
        base_config: Dict,
        kb,
        simulate_fn: SimulateFn,
        max_iterations: int = 4,
        max_candidates_per_expansion: int = 3,
        max_depth: int = 3,
        exploration_constant: float = 1.41,  # sqrt(2): 標準的なUCB1定数
    ):
        self.stage = stage
        self.base_config = base_config
        self.kb = kb
        self.simulate_fn = simulate_fn
        self.max_iterations = max_iterations
        self.max_candidates_per_expansion = max_candidates_per_expansion
        self.max_depth = max_depth
        self.c = exploration_constant

        self.root = MCTSNode(stage=stage, config=base_config, source="root")
        self.iterations_used = 0
        self.last_error_log = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> Dict:
        """
        探索を実行し、以下を返す:
          {
            "success": bool,
            "config": dict,           # 採用された最終パラメータ
            "iterations_used": int,
            "log": str,               # 採用ノードの実行ログ
            "tree": dict,             # to_dict()されたツリー (テレメトリ/可視化用)
          }
        """
        winner = self._simulate_node(self.root, depth=0)
        if winner is not None:
            return self._result(winner, success=True)

        # ルートが失敗 -> 展開して探索開始
        self._expand(self.root, depth=0)

        while self.iterations_used < self.max_iterations and self.root.children:
            node = self._select_leaf_to_visit(self.root)
            if node is None:
                break  # これ以上未実行の枝が無い

            winner = self._simulate_node(node, depth=self._depth_of(node))
            if winner is not None:
                return self._result(winner, success=True)

            # 失敗した場合、深さに余裕があればさらに展開 (連鎖的な修正を試す)
            if not node.children and self._depth_of(node) < self.max_depth:
                self._expand(node, depth=self._depth_of(node))

        # 予算を使い切った場合: 最も見込みのあったノード (Robust Child) を報告用に選ぶ
        best = self._best_attempted_node()
        return self._result(best, success=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _select_leaf_to_visit(self, node: MCTSNode) -> Optional[MCTSNode]:
        """UCB1で未実行の子孫ノードを1つ選ぶ (Selectionフェーズ)。"""
        current = node
        while current.children:
            unvisited = [c for c in current.children if c.effective_visits == 0]
            if unvisited:
                # 事前分布(prior)が最も高いものを優先して実行コストを節約する
                return max(unvisited, key=lambda c: c.value)
            current = current.best_child_by_ucb(self.c)
            if current.visits > 0 and not current.children:
                # 既に実行済みで展開もされていない葉 -> これ以上進めない
                return None
        return None

    def _expand(self, node: MCTSNode, depth: int):
        if depth >= self.max_depth:
            return
        candidates = generate_candidates(
            stage=self.stage,
            base_config=node.config,
            error_log=node.result_log or "",
            kb=self.kb,
            max_candidates=self.max_candidates_per_expansion,
        )
        for cand in candidates:
            child = MCTSNode(
                stage=self.stage,
                config=cand["config"],
                parent=node,
                knowledge_id=cand["knowledge_id"],
                source=cand["source"],
            )
            prior_visits, prior_wins = self.kb.get_prior(cand["knowledge_id"])
            child.set_prior(prior_visits, prior_wins)
            node.children.append(child)

    def _simulate_node(self, node: MCTSNode, depth: int) -> Optional[MCTSNode]:
        """実際にgmxを実行 (本物のプレイアウト)。成功ならそのノードを返す。"""
        if self.iterations_used >= self.max_iterations and node is not self.root:
            return None

        success, reward, log = self.simulate_fn(self.stage, node.config)
        self.iterations_used += 1
        node.result_log = log
        self.last_error_log = log
        node.backpropagate(reward)

        # KnowledgeBaseへ結果を反映 (自己学習)
        if node.knowledge_id is not None:
            self.kb.record_result(node.knowledge_id, success)
        elif node.source == "heuristic":
            # 未知だったヒューリスティック候補が有効だった/無効だったことを新規レコード化
            new_id = self.kb.record_new_result(
                pattern=self._extract_pattern(log),
                cause=f"Auto-discovered during MCTS search at stage={self.stage}",
                solution=str(node.config),
                stage=self.stage,
                param_delta=node.config,
                success=success,
            )
            node.knowledge_id = new_id

        if success:
            node.is_terminal_success = True
            return node
        return None

    def _depth_of(self, node: MCTSNode) -> int:
        depth = 0
        cur = node
        while cur.parent is not None:
            depth += 1
            cur = cur.parent
        return depth

    def _best_attempted_node(self) -> MCTSNode:
        """探索木全体から、最も試行価値の高かったノードを1つ選ぶ (深さ優先で robust child を辿る)。"""
        best = self.root
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n.visits > 0 and (n.value > best.value or best is self.root):
                if n is not self.root:
                    best = n
            stack.extend(n.children)
        return best

    @staticmethod
    def _extract_pattern(log: str) -> str:
        if not log:
            return "unknown error"
        for line in log.splitlines():
            line = line.strip()
            if line:
                return line[:120]
        return "unknown error"

    def _result(self, node: MCTSNode, success: bool) -> Dict:
        return {
            "success": success,
            "config": node.config,
            "iterations_used": self.iterations_used,
            "log": node.result_log or "",
            "tree": self.root.to_dict(),
        }
