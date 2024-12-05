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
            self.state.print_board()
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
                child_node = Node(state=new_state, parent=node)
                node.children.append(child_node)
                return child_node

        print("Expand: No valid moves to expand.")
        return None

    def simulate(self, state):
        current_state = deepcopy(state)

        while not current_state.is_terminal():
            legal_moves = current_state.get_legal_moves()
            if not legal_moves:
                break

            # Prioritize blocking or winning moves
            for move in legal_moves:
                temp_state = deepcopy(current_state)
                temp_state.perform_move(move)
                if temp_state.is_winning_move(*temp_state.last_move):  # Favor critical moves
                    current_state.perform_move(move)
                    break
            else:
                # Default to random move if no critical moves exist
                move = random.choice(legal_moves)
                current_state.perform_move(move)

        # Evaluate the terminal state
        winner = current_state.get_winner()
        if winner == self.player:
            return 1  # Win for the computer
        elif winner:  # Opponent wins
            return -1
        else:  # Draw
            return 0

    def backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.value += result
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
            if not root.state.test_mode:
                print("Tree expansion failed. Falling back to blocking logic.")
            legal_moves = root.state.get_legal_moves()
            if legal_moves:
                # Check for blocking or winning moves
                for move in legal_moves:
                    test_state = deepcopy(root.state)
                    test_state.perform_move(move)
                    if test_state.is_winning_move(*test_state.last_move):
                        if not root.state.test_mode:
                            print(f"Fallback: Blocking or Winning Move Found at Column {move}")
                        return move
                if not root.state.test_mode:
                    # Default to random move if no critical move exists
                    print("Fallback: No blocking or winning move found. Choosing random legal move.")
                return random.choice(legal_moves)
            root.state.print_board()
            raise ValueError("No legal moves available for fallback.")

        return root.best_child(exploration_weight=0).state.last_move[1]