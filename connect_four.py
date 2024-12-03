import numpy as np
import random
from mcts import MCTS, Node
from copy import deepcopy

ROWS = 6
COLS = 7
EMPTY = 0
PLAYER = 1
COMPUTER = 2
COMPUTER_1 = 1
COMPUTER_2 = 2
PLAYER_1 = 1
PLAYER_2 = 2

class CVC_ConnectFour():
    def __init__(self, policy_1: int, policy_2: int):
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.current_player = COMPUTER_1
        self.policies = [policy_1, policy_2]
        self.last_move = None  # Initialize the last_move attribute

    def print_board(self):
        print(np.flip(self.board, 0))
        print("  0 1 2 3 4 5 6")
        print("\n\n")

    def drop_piece(self, col):
        for row in range(ROWS):
            if self.board[row][col] == EMPTY:
                self.board[row][col] = self.current_player
                self.last_move = (row, col)  # Update last_move here
                return row, col
        return None

    def is_winning_move(self, row, col):
        result = (self.check_direction(row, col, 1, 0) or  # Horizontal
                self.check_direction(row, col, 0, 1) or  # Vertical
                self.check_direction(row, col, 1, 1) or  # Diagonal /
                self.check_direction(row, col, 1, -1))   # Diagonal \
        return result
        

    def check_direction(self, row, col, delta_row, delta_col):
        count = 0
        for direction in [1, -1]:
            for step in range(1, 5):
                r = row + direction * step * delta_row
                c = col + direction * step * delta_col
                if 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == self.current_player:
                    count += 1
                else:
                    break
        return count >= 3

    def is_full(self):
        return all(self.board[ROWS - 1, :] != EMPTY)

    def copy(self):
        new_game = CVC_ConnectFour(self.policies[0], self.policies[1])
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game

    def get_legal_moves(self):
        return [col for col in range(COLS) if self.board[ROWS - 1][col] == EMPTY]

    def perform_move(self, col):
        if col not in self.get_legal_moves():
            raise ValueError(f"Invalid move {col}")
        for row in range(ROWS):
            if self.board[row][col] == EMPTY:
                self.board[row][col] = self.current_player
                self.last_move = (row, col)  # Update last_move here
                self.current_player = COMPUTER_2 if self.current_player == COMPUTER_1 else COMPUTER_1
                return

    def is_terminal(self):
        if self.get_winner() is not None or not self.get_legal_moves():
            return True
        return False

    def get_winner(self):
        for col in range(COLS):
            for row in range(ROWS):
                if self.board[row][col] != EMPTY and self.is_winning_move(row, col):
                    return self.board[row][col]
        return None

    def computer_move(self):
        if self.current_player == COMPUTER_1:
            policy = self.policies[0]
        else:
            policy = self.policies[1]
        
        if policy == 1:
            # Monte Carlo Tree Search
            for col in self.get_legal_moves():
                test_state = deepcopy(self)
                test_state.perform_move(col)
                if test_state.is_winning_move(*test_state.last_move):  # Prevent opponent win
                    return col

            # Monte Carlo Tree Search
            mcts_bot = MCTS(player=self.current_player)
            root = Node(state=self)
            move = mcts_bot.best_move(root, simulations=1000)
            if not isinstance(move, int):
                raise ValueError(f"Invalid move returned: {move}")
            if move is None:
                # Fallback: Block if possible
                for col in self.get_legal_moves():
                    test_state = deepcopy(self)
                    test_state.perform_move(col)
                    if test_state.is_winning_move(*test_state.last_move):
                        return col
                # Default: Choose any legal move
                return random.choice(self.get_legal_moves())
            return move
        
        elif policy == 2:
            # Simple heuristic: First available column
            for col in range(COLS):
                if self.board[ROWS-1][col] == EMPTY:
                    return col
            return None
        
        elif policy == 3:
            # JASMINE TO DO
            # Advanced heuristic: Add to biggest available group of player's chips already placed
            raise Exception("JASMINE TO DO: Advanced heuristic")

        elif policy == 4:
            # Random open column
            available = []
            for col in range(COLS):
                if self.board[ROWS-1][col] == EMPTY:
                    available.append(col)
            
            if not available:
                return None
            else:
                return random.choice(available)
        
        else:
            raise Exception("Error: please enter a valid computer strategy / policy for ", self.current_player)

    def play_game(self):
        while True:
            self.print_board()
            if self.current_player == COMPUTER_1:
                col = self.computer_move()
                print(f"Computer 1 chooses column: {col}")
            else:
                col = self.computer_move()
                print(f"Computer 2 chooses column: {col}")

            if 0 <= col < COLS and self.board[ROWS-1][col] == EMPTY:
                row, col = self.drop_piece(col)

                if self.is_winning_move(row, col):
                    self.print_board()
                    if self.current_player == COMPUTER_1:
                        print("Computer 1 wins!")
                    else:
                        print("Computer 2 wins!")
                    break

                self.current_player = COMPUTER_2 if self.current_player == COMPUTER_1 else COMPUTER_1

            if self.is_full():
                self.print_board()
                print("It's a draw!")
                break

class PVC_ConnectFour():
    def __init__(self, policy : int):
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.current_player = PLAYER
        self.policy = policy

    def print_board(self):
        print(np.flip(self.board, 0))
        print("  0 1 2 3 4 5 6")
        print("\n\n")

    def drop_piece(self, col):
        for row in range(ROWS):
            if self.board[row][col] == EMPTY:
                self.board[row][col] = self.current_player
                return row, col
        return None

    def is_winning_move(self, row, col):
        # Check horizontal, vertical, and diagonal connections
        return (self.check_direction(row, col, 1, 0) or  # Horizontal
                self.check_direction(row, col, 0, 1) or  # Vertical
                self.check_direction(row, col, 1, 1) or  # Diagonal /
                self.check_direction(row, col, 1, -1))   # Diagonal \

    def check_direction(self, row, col, delta_row, delta_col):
        count = 0
        for direction in [1, -1]:
            for step in range(1, 5):
                r = row + direction * step * delta_row
                c = col + direction * step * delta_col
                if 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == self.current_player:
                    count += 1
                else:
                    break
        return count >= 3

    def is_full(self):
        return all(self.board[ROWS - 1, :] != EMPTY)

    def copy(self):
        new_game = CVC_ConnectFour(self.policies[0], self.policies[1])
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game

    def get_legal_moves(self):
        return [col for col in range(COLS) if self.board[ROWS - 1][col] == EMPTY]

    def perform_move(self, col):
        for row in range(ROWS):
            if self.board[row][col] == EMPTY:
                self.board[row][col] = self.current_player
                self.last_move = (row, col)  # Update last_move here
                self.current_player = COMPUTER_2 if self.current_player == COMPUTER_1 else COMPUTER_1
                return

    def is_terminal(self):
        if self.is_full():
            return True
        for col in range(COLS):
            for row in range(ROWS):
                if self.board[row][col] != EMPTY and self.is_winning_move(row, col):
                    return True
        return False

    def get_winner(self):
        for col in range(COLS):
            for row in range(ROWS):
                if self.board[row][col] != EMPTY and self.is_winning_move(row, col):
                    return self.board[row][col]
        return None

    def computer_move(self):
        policy = self.policy
        
        if policy == 1:
            # Monte Carlo Tree Search
            for col in self.get_legal_moves():
                test_state = deepcopy(self)
                test_state.perform_move(col)
                if test_state.is_winning_move(*test_state.last_move):  # Prevent opponent win
                    return col

            # Monte Carlo Tree Search
            mcts_bot = MCTS(player=self.current_player)
            root = Node(state=self)
            move = mcts_bot.best_move(root, simulations=1000)
            if not isinstance(move, int):
                raise ValueError(f"Invalid move returned: {move}")
            if move is None:
                # Fallback: Block if possible
                for col in self.get_legal_moves():
                    test_state = deepcopy(self)
                    test_state.perform_move(col)
                    if test_state.is_winning_move(*test_state.last_move):
                        return col
                # Default: Choose any legal move
                return random.choice(self.get_legal_moves())
            return move
        
        elif policy == 2:
            # Simple Heuristic: Choose the first available column
            for col in range(COLS):
                if self.board[ROWS-1][col] == EMPTY:
                    return col
            return None
        
        elif policy == 3:
            # JASMINE TO DO
            # Advanced Heuristic: Add to biggest available group of player's chips already placed
            raise Exception("JASMINE TO DO: Advanced heuristic")

        elif policy == 4:
            # Random Agent: Choose a random available column
            available = []
            for col in range(COLS):
                if self.board[ROWS-1][col] == EMPTY:
                    available.append(col)
            
            if not available:
                return None
            else:
                return random.choice(available)
        
        else:
            raise Exception("Error: please enter a valid computer strategy / policy for ", self.current_player)

    def play_game(self):
        while True:
            self.print_board()
            if self.current_player == PLAYER:
                col = int(input("Choose a column (0-6): "))
            else:
                col = self.computer_move()
                print(f"Computer chooses column: {col}")

            if 0 <= col < COLS and self.board[ROWS-1][col] == EMPTY:
                row, col = self.drop_piece(col)

                if self.is_winning_move(row, col):
                    self.print_board()
                    if self.current_player == PLAYER:
                        print("You win!")
                    else:
                        print("Computer wins!")
                    break

                self.current_player = COMPUTER if self.current_player == PLAYER else PLAYER

            if self.is_full():
                self.print_board()
                print("It's a draw!")
                break

class PVP_ConnectFour():
    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.current_player = PLAYER_1

    def print_board(self):
        print(np.flip(self.board, 0))
        print("  0 1 2 3 4 5 6")
        print("\n\n")

    def drop_piece(self, col):
        for row in range(ROWS):
            if self.board[row][col] == EMPTY:
                self.board[row][col] = self.current_player
                return row, col
        return None

    def is_winning_move(self, row, col):
        # Check horizontal, vertical, and diagonal connections
        return (self.check_direction(row, col, 1, 0) or  # Horizontal
                self.check_direction(row, col, 0, 1) or  # Vertical
                self.check_direction(row, col, 1, 1) or  # Diagonal /
                self.check_direction(row, col, 1, -1))   # Diagonal \

    def check_direction(self, row, col, delta_row, delta_col):
        count = 0
        for direction in [1, -1]:
            for step in range(1, 5):
                r = row + direction * step * delta_row
                c = col + direction * step * delta_col
                if 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == self.current_player:
                    count += 1
                else:
                    break
        return count >= 3

    def is_full(self):
        return all(self.board[ROWS-1, :] != EMPTY)

    def play_game(self):
        while True:
            self.print_board()
            if self.current_player == PLAYER_1:
                col = int(input("Player 1, please choose a column (0-6): "))
            else:
                col = int(input("Player 2, please choose a column (0-6): "))

            if 0 <= col < COLS and self.board[ROWS-1][col] == EMPTY:
                row, col = self.drop_piece(col)

                if self.is_winning_move(row, col):
                    self.print_board()
                    if self.current_player == PLAYER_1:
                        print("Player 1 wins!")
                    else:
                        print("Player 2 wins!")
                    break

                self.current_player = PLAYER_2 if self.current_player == PLAYER_1 else PLAYER_1

            if self.is_full():
                self.print_board()
                print("It's a draw!")
                break

def main():
    mode = str(input("\n 1) Enter 'cvc' or 1 to run a computer vs. computer game \n 2) Enter 'pvc' or 2 to run a player vs. computer game \n 3) Enter 'pvp' or 3 to run a player vs. player game \n\n Input: "))

    if mode == "cvc" or mode == "1":
        policy_1 = int(input("\n\n What strategy / policy would you like computer 1 to use? \n Enter the number corresponding to the desired strategy: \n 1) Monte Carlo Tree Search \n 2) Simple Heuristic \n 3) Adanvanced Heuristic \n 4) Random \n\n Input: "))
        policy_2 = int(input("\n\n What strategy / policy would you like compueter 2 to use? \n Enter the number corresponding to the desired strategy: \n 1) Monte Carlo Tree Search \n 2) Simple Heuristic \n 3) Adanvanced Heuristic \n 4) Random \n\n Input: "))
        game = CVC_ConnectFour(policy_1, policy_2)
    elif mode == "pvc" or mode == "2":
        policy = int(input("\n\nWhat strategy / policy would you like the computer to use? \n Enter the number corresponding to the desired strategy: \n 1) Monte Carlo Tree Search \n 2) Simple Heuristic \n 3) Adanvanced Heuristic \n 4) Random \n\n Input: "))
        game = PVC_ConnectFour(policy)
    elif mode == "pvp" or mode == "3":
        game = PVP_ConnectFour()
    else:
        raise Exception("Error: please enter a valid game mode")
    
    print("\n\n")
    game.play_game()

if __name__ == "__main__":
    main()

# JASMINE TO DO: CREATE FRAMEWORK TO RUN 1000 CVC GAMES
# USING A SPECIFIED STRATEGY FOR EACH COMPUTER
# TRACK THE NUMBER OF TIMES EACH COMUTER WINS, THEIR APPROACH,
# AND THE NUMBER OF MOVES IT TOOK THEM TO WIN!
# (e.g. add a num_moves class variable and return num_moves and the winner from play_game)
# call, for example, CVC_ConnectFour(1, 3) 1000 times tracking num_moves and winner, 
# with comp1 running MCTS and comp2 running the advanced heuristic
# add if statements before print statements so that they do not execute in test mode