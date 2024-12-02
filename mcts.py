import random
import numpy as np
from copy import deepcopy

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_moves())

    def best_child(self, exploration_weight=1.0):
        if not self.children:
            raise ValueError("No children to select in `best_child`. Tree expansion likely failed.")

        # Check for immediate winning moves
        for child in self.children:
            if child.state.is_terminal() and child.state.get_winner() == self.player:
                return child

        weights = [
            (child.value / (child.visits + 1e-6)) +
            exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6))
            for child in self.children
        ]
        return self.children[np.argmax(weights)]

class MCTS:
    def __init__(self, player):
        self.player = player

    def select(self, node):
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = node.best_child()
        return node

    def expand(self, node):
        legal_moves = node.state.get_legal_moves()
        tried_moves = [child.state.last_move[1] for child in node.children if child.state.last_move]

        # print(f"Expanding Node with Board State:\n{node.state.board}")
        # print(f"Legal Moves: {legal_moves}, Tried Moves: {tried_moves}")
        for move in legal_moves:
            if move not in tried_moves:
                new_state = deepcopy(node.state)
                new_state.perform_move(move)
                if new_state.is_terminal():  # Skip adding terminal nodes
                    # print(f"Skipping terminal state for move: {move}")
                    continue
                child_node = Node(state=new_state, parent=node)
                node.children.append(child_node)
                # print(f"Expanded new child for move: {move}")
                return child_node
        # print(f"No new children could be expanded for node with state:\n{node.state.board}")
        return None

    def simulate(self, state):
        current_state = deepcopy(state)
        seen_moves = set()  # Track moves to discourage repetitive moves during simulation

        while not current_state.is_terminal():
            legal_moves = current_state.get_legal_moves()

            if not legal_moves:
                break

            move_scores = []
            for move in legal_moves:
                temp_state = deepcopy(current_state)
                temp_state.perform_move(move)

                # Reward immediate wins
                if temp_state.is_winning_move(*temp_state.last_move):
                    move_scores.append((move, 1.0))
                # Reward blocking opponent wins
                elif current_state.current_player != self.player and temp_state.is_winning_move(*temp_state.last_move):
                    move_scores.append((move, 0.9))
                # Penalize stacking in already-seen moves
                elif move in seen_moves:
                    move_scores.append((move, 0.1))
                # Encourage center columns
                elif 2 <= move <= 4:
                    move_scores.append((move, 0.6))
                else:
                    move_scores.append((move, 0.3))

            # Check if move_scores has valid positive weights
            if not move_scores or all(score <= 0 for _, score in move_scores):
                raise ValueError("No valid moves with positive weights during simulation.")

            moves, scores = zip(*move_scores)
            move = random.choices(moves, weights=scores, k=1)[0]
            current_state.perform_move(move)
            seen_moves.add(move)

        # Evaluate game result
        winner = current_state.get_winner()
        reward = 1 if winner == self.player else (-1 if winner else 0)
        # print(f"Simulate: Winner = {winner}, Reward = {reward}")
        return reward


    def backpropagate(self, node, result):
        diversity_bonus = 0.1  # Reward for exploring new paths
        while node is not None:
            node.visits += 1
            node.value += result + diversity_bonus
            result = -result  # Reverse result for opponent
            node = node.parent

    def best_move(self, root, simulations=1000):
        for _ in range(simulations):
            leaf = self.select(root)
            if leaf is None:
                continue
            result = self.simulate(leaf.state)
            self.backpropagate(leaf, result)

        if not root.children:
            print("Root Node Debug Info:")
            print(f"Root Board State:\n{root.state.board}")
            print(f"Legal Moves at Root: {root.state.get_legal_moves()}")
            print(f"Is Root Terminal: {root.state.is_terminal()}")
            raise ValueError("Tree expansion failed. Root has no children.")


        return root.best_child(exploration_weight=0).state.last_move[1]