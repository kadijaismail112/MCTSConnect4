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
                # Expand and return a new node, or exit loop if no expansion is possible
                expanded_node = self.expand(node)
                if expanded_node:
                    return expanded_node
                break  # Exit if no expansion is possible
            else:
                node = node.best_child()
        return node

    def expand(self, node):
        legal_moves = node.state.get_legal_moves()
        tried_moves = [child.state.last_move[1] for child in node.children if child.state.last_move]

        for move in legal_moves:
            if move not in tried_moves:
                new_state = deepcopy(node.state)
                new_state.perform_move(move)
                # Add the node even if it blocks the opponent's win
                child_node = Node(state=new_state, parent=node)
                node.children.append(child_node)
                return child_node

        return None

    def simulate(self, state):
        current_state = deepcopy(state)

        while not current_state.is_terminal():
            legal_moves = current_state.get_legal_moves()
            if not legal_moves:
                break

            # Check for blocking moves
            for move in legal_moves:
                temp_state = deepcopy(current_state)
                temp_state.perform_move(move)
                if temp_state.is_winning_move(*temp_state.last_move):  # Prevent opponent win
                    current_state.perform_move(move)
                    break
            else:  # If no blocking move, pick randomly
                move = random.choice(legal_moves)
                current_state.perform_move(move)

        winner = current_state.get_winner()
        return 1 if winner == self.player else -1 if winner else 0

    def backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.value += result
            result = -result  # Reverse result for opponent
            node = node.parent

    def best_move(self, root, simulations=1000):
        for _ in range(simulations):
            leaf = self.select(root)
            if leaf is None:  # Skip simulation if no leaf is found
                continue
            result = self.simulate(leaf.state)
            self.backpropagate(leaf, result)

        if not root.children:
            # Fallback: Randomly select from legal moves if tree expansion fails
            print("Tree expansion failed. Falling back to random legal move.")
            legal_moves = root.state.get_legal_moves()
            if legal_moves:
                return random.choice(legal_moves)
            raise ValueError("No legal moves available for fallback.")

        return root.best_child(exploration_weight=0).state.last_move[1]