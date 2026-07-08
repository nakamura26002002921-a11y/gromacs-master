# src/gromacs_agent/search/mcts.py
import math
import random
from typing import Optional, List, Dict
from gromacs_agent.tools.gromacs_tools import GromacsTools
from gromacs_agent.memory.knowledge_base import KnowledgeBase
from gromacs_agent.llm.multi_fixer import generate_multiple_fixes

class MCTSNode:
    def __init__(self, state: Dict, parent: Optional['MCTSNode'] = None, action: Dict = None):
        self.state = state          # {"step": "em", "args": [...], "config": {...}}
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.reward = 0.0
        self.action = action        # LLMが提案した修正内容

class MCTS:
    def __init__(self, tools: GromacsTools, kb: KnowledgeBase, max_iterations: int = 5, exploration_const: float = 1.41):
        self.tools = tools
        self.kb = kb
        self.max_iters = max_iterations
        self.C = exploration_const  # UCT式の探索係数

    def search(self, initial_state: Dict) -> Optional[Dict]:
        """MCTSを実行し、最も有望な修正アクションを返す"""
        root = MCTSNode(initial_state)
        
        for _ in range(self.max_iters):
            node = self._select(root)
            
            if not self._is_terminal(node):
                node = self._expand(node)
                
            reward = self._simulate(node)
            self._backpropagate(node, reward)
            
        # 最も訪問回数と報酬が高い子ノードのアクションを返す
        if not root.children:
            return None
            
        best_child = max(root.children, key=lambda c: c.reward / max(c.visits, 1))
        return best_child.action

    def _select(self, node: MCTSNode) -> MCTSNode:
        """UCT式に基づいて最も有望なノードを選択"""
        while node.children:
            if node.visits == 0:
                return node
            # UCT (Upper Confidence bound applied to Trees)
            node = max(node.children, key=lambda c: (c.reward / c.visits) + self.C * math.sqrt(math.log(node.visits) / c.visits))
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """LLMを使って子ノード（修正候補）を生成"""
        stage = node.state["step"]
        args = node.state["args"]
        stderr = node.state.get("stderr", "")
        
        # KBから類似事例を取得
        kb_suggestions = self.kb.search_similar_fixes(stage, stderr)
        
        # LLMで複数分岐を生成
        candidates = generate_multiple_fixes(stage, args, stderr, kb_suggestions)
        
        for cand in candidates:
            child_state = {**node.state, "args": cand["args"], "reason": cand["reason"]}
            child_node = MCTSNode(child_state, parent=node, action=cand)
            node.children.append(child_node)
            
        # 未展開の子ノードをランダムに1つ選んで返す
        return random.choice(node.children) if node.children else node

    def _simulate(self, node: MCTSNode) -> float:
        """
        実際にコマンドを実行して報酬を計算する。
        ※ MCTSの計算コストを抑えるため、mdrunではなく grompp (検証のみ) を実行することを推奨
        """
        stage = node.state["step"]
        args = node.state["args"]
        
        # 【重要】mdrunを回すと時間がかかるため、ここではgromppでトポロジーとmdpの整合性チェックのみ行う
        # 実際のmdrunは最終候補が決まってから1回だけ回す
        check_args = self._convert_to_grompp_check(stage, args)
        
        code, stdout, stderr = self.tools.run_gmx_command("grompp", check_args, cwd=node.state.get("work_dir", "."))
        
        node.state["stderr"] = stderr # エラーメッセージを記録
        
        if code == 0:
            return 1.0  # 成功（高い報酬）
        else:
            return 0.0  # 失敗（低い報酬）

    def _convert_to_grompp_check(self, stage: str, args: list) -> list:
        """mdrunの引数をgromppの検証用引数に変換するヘルパー"""
        # 例: em ステージなら、em.mdp と solvated.gro を使って grompp -f em.mdp -c solvated.gro -p topol.top -o em.tpr を実行
        # 実際の実装ではステージごとのマッピングテーブルを持つ
        return ["-f", f"{stage}.mdp", "-c", "solvated.gro", "-p", "topol.top", "-o", f"{stage}_check.tpr", "-maxwarn", "1"]

    def _backpropagate(self, node: MCTSNode, reward: float):
        """葉ノードからルートへ報酬を伝播"""
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def _is_terminal(self, node: MCTSNode) -> bool:
        return len(node.children) > 0 or node.state.get("is_success", False)
