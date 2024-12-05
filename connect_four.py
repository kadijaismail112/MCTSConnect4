import numpy as np
import random
from matplotlib import pyplot as plt
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
    def __init__(self, policy_1: int, policy_2: int, test: bool):
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.current_player = random.choice([COMPUTER_1, COMPUTER_2]) # randomize starting player
        self.policies = [policy_1, policy_2]
        self.last_move = None  # Initialize the last_move attribute
        self.test_winner = None
        self.test_mode = test
        self.num_moves_computer_1 = 0
        self.num_moves_computer_2 = 0

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
                if self.test_winner is not None:
                    if 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == self.test_winner:
                        count += 1
                    else:
                        break
                else: 
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
            if self.current_player == COMPUTER_1:
                computer = COMPUTER_1
                player = COMPUTER_2
            else:
                computer = COMPUTER_2
                player = COMPUTER_1
            def block_opponent_win():
                """
                Check if the opponent is one move away from winning and prioritize blocking those moves.
                """
                best_block_move = None
                for row in range(ROWS):
                    for col in range(COLS):
                        if self.board[row][col] == EMPTY:
                            self.test_winner = player
                            self.board[row][col] = self.test_winner
                            if self.is_winning_move(row, col):  # player has a winning move
                                if (((row - 1) >= 0 and (row - 1) != EMPTY) or (row == 0)):
                                    best_block_move = col
                            self.board[row][col] = EMPTY
                            self.test_winner = None
                return best_block_move

            # Advanced Heuristic: Create or expand the largest viable group of the computer's chips.
            def is_viable_group(group, direction):
                """
                Check if a given group of connected chips can potentially become a complete group of 4.
                The group should have at least 2 connected chips and should have space in some direction
                to grow into a full group of 4.
                """
                if len(group) == 1:
                    # Directions to check: horizontal, vertical, diagonal (/), diagonal (\)
                    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Right, Down, Down-right, Down-left
                    
                    # For each direction, check if the group can be expanded into a full group of 4
                    for dr, dc in directions:
                        count_chip = 0  # Number of already connected chips in this direction
                        count_empty = 0  # Number of empty spaces that could be filled to form a group of 4

                        # We check both directions from the group (left-right, up-down, diagonal)
                        for step in [-1, 1]:  # Check both directions: -1 (left/up), 1 (right/down)
                            for i in range(0, 4):  # Check up to 3 positions in this direction in addition to the current position
                                for r,c in group:
                                    nr = r + dr * i * step
                                    nc = c + dc * i * step
                                    if 0 <= nr < ROWS and 0 <= nc < COLS:
                                        if self.board[nr][nc] == computer:
                                            count_chip += 1
                                        elif self.board[nr][nc] == EMPTY:
                                            count_empty += 1

                        # If the total number of chips + empty spaces is at least 4, return True
                        if count_chip + count_empty >= 4:
                            return True
                    return False
                else: # groups of 2 or more have a direction
                    dr = direction[0]
                    dc = direction[1]

                    count_chip = 0  # Number of already connected chips in this direction
                    count_empty = 0  # Number of empty spaces that could be filled to form a group of 4

                    # We check both directions from the group (left-right, up-down, diagonal)
                    for step in [-1, 1]:  # Check both directions: -1 (left/up), 1 (right/down)
                        for i in range(0, 4):  # Check up to 3 positions in this direction in addition to the current position
                            for r,c in group:
                                nr = r + dr * i * step
                                nc = c + dc * i * step
                                if 0 <= nr < ROWS and 0 <= nc < COLS:
                                    if self.board[nr][nc] == computer:
                                        count_chip += 1
                                    elif self.board[nr][nc] == EMPTY:
                                        count_empty += 1

                    # If the total number of chips + empty spaces in the desired direction is at least 4, return True
                    if count_chip + count_empty >= 4:
                        return True
                return False

            def find_connected_groups(row, col):
                """
                Finds all connected groups of 2 or more chips for the computer, including infilling
                gaps of one empty space between chips that would form a valid group after the current move.
                """
                # Directions to check: horizontal, vertical, diagonal (/), diagonal (\)
                directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Right, Down, Down-right, Down-left
                connected_groups = []

                # We'll check all possible directions for groups of 2 or more connected chips
                for dr, dc in directions:
                    connected_group = set()  # To store all coordinates of the connected group
                    for i in range(-1, 2):  # Check 3 positions: one to the left, current position, one to the right
                        group = set()
                        empty_spaces = set()

                        # Look at the current position and neighboring positions in the direction of interest
                        for step in range(-1, 2):  # Step -1, 0, 1 (previous, current, next)
                            nr = row + dr * step
                            nc = col + dc * step

                            if 0 <= nr < ROWS and 0 <= nc < COLS:
                                # Check the current position
                                if self.board[nr][nc] == computer:
                                    group.add((nr, nc))
                                elif self.board[nr][nc] == EMPTY:
                                    empty_spaces.add((nr, nc))

                        # If the group is valid and contains two or more connected chips
                        if len(group) >= 1:
                            connected_group.update(group)

                        # Now we look for valid opportunities to fill a single gap (one empty space) between chips
                        if len(group) == 1 and len(empty_spaces) == 1:
                            # We have one chip and one empty space between chips
                            # We check if placing a new chip in the empty space would create a connected group
                            (er, ec) = empty_spaces.pop()  # Get the empty space
                            '''if is_viable_group(group, [dr, dc]):  # Check if this potential group will form a group of 4
                                connected_group.add((er, ec))
                            else:
                                return None  # No valid group after this move'''
                            if not is_viable_group(group, [dr, dc]):  # Check if this potential group will form a group of 4
                                return None  # No valid group after this move
                    
                    '''if len(connected_group) >= 2:'''
                    if len(connected_group) >= 1:
                        connected_group_with_dir = [connected_group, [dr, dc]]
                        connected_groups.append(connected_group_with_dir)  # add the found connected group

                if connected_groups:
                    return connected_groups
                else:
                    return None  # No valid group found

            def find_best_move_for_group(group_with_dir):
                """
                Find the best move for expanding or infilling the largest connected group.
                The function checks for gaps that can be infilled, or places a chip in a direction
                that expands the group to potentially form a line of 4.
                """
                if group_with_dir is None:
                    # If no group can be infilled or expanded, fall back to random selection 
                    # Or available-row-based selection.
                    # Want to choose randomly between placing a random chip
                    # Or placing a chip in the column(s) with the most rows available.
                    options = [1, 2]
                    chosen_option = random.choice(options)
                    best_columns = []

                    if chosen_option == 1:
                        # choose a random column
                        for i in range(COLS):
                            best_columns.append(i)
                    else:
                        # choose the column with the most rows available
                        available_moves = self.get_legal_moves()
                        if not available_moves:
                            return None  # No moves available.

                        # Find columns with the most available rows.
                        max_rows_available = 0
                        for col in available_moves:
                            rows_available = sum(1 for row in range(ROWS) if self.board[row][col] == EMPTY)
                            if rows_available > max_rows_available:
                                max_rows_available = rows_available
                                best_columns = [col]
                            elif rows_available == max_rows_available:
                                best_columns.append(col)
                    
                    # Pick randomly (or randomly from the columns with the most available rows).
                    return random.choice(best_columns)

                else: 
                    best_move_for_group = None
                    largest_group_size = 0
                    group = group_with_dir[0]
                    direction = group_with_dir[1]

                    # If the group is size 1, iterate through the directions to find the best move that infills or expands the group
                    if len(group) == 1:
                        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                            for r, c in group:
                                # Try expanding in both directions (left-right, up-down, etc.)
                                for step in [-1, 1]:
                                    nr, nc = r, c
                                    for i in range(0, 4):  # Expand 3 steps in this direction including current space
                                        nr += dr * step
                                        nc += dc * step
                                        
                                        # Check if it's a valid move
                                        if 0 <= nr < ROWS and 0 <= nc < COLS:
                                            if self.board[nr][nc] == EMPTY:
                                                # Check if placing here would form a valid group of 4
                                                group_copy = group.copy()
                                                group_copy.add((nr, nc))
                                                if is_viable_group(group_copy, direction):
                                                    # If it's a viable group, consider this as a potential best move
                                                    if len(group_copy) > largest_group_size:
                                                        largest_group_size = len(group_copy)
                                                        best_move_for_group = nc  # Set the column as the best move
                    
                    # Otherwise, expand the group in its direction
                    else:
                        dr = direction[0]
                        dc = direction[1]
                        for r, c in group:
                            # Try expanding in both directions (left-right, up-down, etc.)
                            for step in [-1, 1]:
                                nr, nc = r, c
                                for i in range(0, 4):  # Expand 3 steps in this direction including current space
                                    nr += dr * step
                                    nc += dc * step
                                    
                                    # Check if it's a valid move
                                    if 0 <= nr < ROWS and 0 <= nc < COLS:
                                        if self.board[nr][nc] == EMPTY:
                                            # Check if placing here would form a valid group of 4
                                            group_copy = group.copy()
                                            group_copy.add((nr, nc))
                                            if is_viable_group(group_copy, direction):
                                                # If it's a viable group, consider this as a potential best move
                                                if len(group_copy) > largest_group_size:
                                                    largest_group_size = len(group_copy)
                                                    best_move_for_group = nc  # Set the column as the best move
                    

                    return best_move_for_group

            if block_opponent_win() is None:
                best_moves = []
                largest_group_size = 0
            
                # Check all possible groups of connected chips on the board.
                # For each group, check if expanding it would be viable.
                for col in range(COLS):
                    for row in range(ROWS):
                        if self.board[row][col] == computer:
                            # Look for a group of connected chips that can be expanded
                            groups = find_connected_groups(row, col)  # A helper function to find the group
                            
                            if groups:
                                for group_with_dir in groups:
                                    group = group_with_dir[0]
                                    best_move_for_group = find_best_move_for_group(group_with_dir)

                                    if best_move_for_group is not None:

                                        added_row = -1

                                        # getting the row and col of the best_move_for_group
                                        for r in range(ROWS):
                                            if self.board[r][best_move_for_group] == EMPTY:
                                                added_row = r
                                                break

                                        group_copy = group.copy()
                                        group_copy.add((added_row, best_move_for_group))

                                        if len(group_copy) > largest_group_size:
                                            largest_group_size = len(group_copy)
                                            best_moves = [best_move_for_group]  # Set the column as the best move
                                        elif len(group_copy) == largest_group_size:
                                            best_moves.append(best_move_for_group)
                if not best_moves:
                    return find_best_move_for_group(None) # no chips placed yet
                
                else:
                    # In the event of a tie, pick a random best move from the options
                    return random.choice(best_moves)
            else: 
                #print("blocked")
                return block_opponent_win()

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

            if self.is_full():
                if not self.test_mode:
                    self.print_board()
                    print("It's a draw!")
                    #break
                return "d"
            
            if not self.test_mode:
                self.print_board()
            
            if self.current_player == COMPUTER_1:
                col = self.computer_move()
                if not self.test_mode:
                    print(f"Computer 1 chooses column: {col}")
            else:
                col = self.computer_move()
                if not self.test_mode:
                    print(f"Computer 2 chooses column: {col}")
            
            if col is None and self.test_mode:
                col = random.choice(self.get_legal_moves())

            if 0 <= col < COLS and self.board[ROWS-1][col] == EMPTY:
                row, col = self.drop_piece(col)
                
                if self.current_player == COMPUTER_1:
                    self.num_moves_computer_1 += 1
                else:
                    self.num_moves_computer_2 += 1

                if self.is_winning_move(row, col):
                    if not self.test_mode:
                        self.print_board()
                        if self.current_player == COMPUTER_1:
                            print("Computer 1 wins!")
                        else:
                            print("Computer 2 wins!")
                        #break
                    return str(self.current_player)

                self.current_player = COMPUTER_2 if self.current_player == COMPUTER_1 else COMPUTER_1

class PVC_ConnectFour():
    def __init__(self, policy : int):
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.current_player = random.choice([PLAYER, COMPUTER]) # randomize starting player
        self.policy = policy
        self.test_winner = None

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
                if self.test_winner is not None:
                    if 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == self.test_winner:
                        count += 1
                    else:
                        break
                else: 
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
            def block_opponent_win():
                """
                Check if the opponent is one move away from winning and prioritize blocking those moves.
                """
                best_block_move = None
                for row in range(ROWS):
                    for col in range(COLS):
                        if self.board[row][col] == EMPTY:
                            self.test_winner = PLAYER
                            self.board[row][col] = self.test_winner
                            if self.is_winning_move(row, col):  # player has a winning move
                                if (((row - 1) >= 0 and (row - 1) != EMPTY) or (row == 0)):
                                    best_block_move = col
                            self.board[row][col] = EMPTY
                            self.test_winner = None
                return best_block_move

            # Advanced Heuristic: Create or expand the largest viable group of the computer's chips.
            def is_viable_group(group, direction):
                """
                Check if a given group of connected chips can potentially become a complete group of 4.
                The group should have at least 2 connected chips and should have space in some direction
                to grow into a full group of 4.
                """
                if len(group) == 1:
                    # Directions to check: horizontal, vertical, diagonal (/), diagonal (\)
                    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Right, Down, Down-right, Down-left
                    
                    # For each direction, check if the group can be expanded into a full group of 4
                    for dr, dc in directions:
                        count_chip = 0  # Number of already connected chips in this direction
                        count_empty = 0  # Number of empty spaces that could be filled to form a group of 4

                        # We check both directions from the group (left-right, up-down, diagonal)
                        for step in [-1, 1]:  # Check both directions: -1 (left/up), 1 (right/down)
                            for i in range(0, 4):  # Check up to 3 positions in this direction in addition to the current position
                                for r,c in group:
                                    nr = r + dr * i * step
                                    nc = c + dc * i * step
                                    if 0 <= nr < ROWS and 0 <= nc < COLS:
                                        if self.board[nr][nc] == COMPUTER:
                                            count_chip += 1
                                        elif self.board[nr][nc] == EMPTY:
                                            count_empty += 1

                        # If the total number of chips + empty spaces is at least 4, return True
                        if count_chip + count_empty >= 4:
                            return True
                    return False
                else: # groups of 2 or more have a direction
                    dr = direction[0]
                    dc = direction[1]

                    count_chip = 0  # Number of already connected chips in this direction
                    count_empty = 0  # Number of empty spaces that could be filled to form a group of 4

                    # We check both directions from the group (left-right, up-down, diagonal)
                    for step in [-1, 1]:  # Check both directions: -1 (left/up), 1 (right/down)
                        for i in range(0, 4):  # Check up to 3 positions in this direction in addition to the current position
                            for r,c in group:
                                nr = r + dr * i * step
                                nc = c + dc * i * step
                                if 0 <= nr < ROWS and 0 <= nc < COLS:
                                    if self.board[nr][nc] == COMPUTER:
                                        count_chip += 1
                                    elif self.board[nr][nc] == EMPTY:
                                        count_empty += 1

                    # If the total number of chips + empty spaces in the desired direction is at least 4, return True
                    if count_chip + count_empty >= 4:
                        return True
                return False

            def find_connected_groups(row, col):
                """
                Finds all connected groups of 2 or more chips for the computer, including infilling
                gaps of one empty space between chips that would form a valid group after the current move.
                """
                # Directions to check: horizontal, vertical, diagonal (/), diagonal (\)
                directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Right, Down, Down-right, Down-left
                connected_groups = []

                # We'll check all possible directions for groups of 2 or more connected chips
                for dr, dc in directions:
                    connected_group = set()  # To store all coordinates of the connected group
                    for i in range(-1, 2):  # Check 3 positions: one to the left, current position, one to the right
                        group = set()
                        empty_spaces = set()

                        # Look at the current position and neighboring positions in the direction of interest
                        for step in range(-1, 2):  # Step -1, 0, 1 (previous, current, next)
                            nr = row + dr * step
                            nc = col + dc * step

                            if 0 <= nr < ROWS and 0 <= nc < COLS:
                                # Check the current position
                                if self.board[nr][nc] == COMPUTER:
                                    group.add((nr, nc))
                                elif self.board[nr][nc] == EMPTY:
                                    empty_spaces.add((nr, nc))

                        # If the group is valid and contains two or more connected chips
                        if len(group) >= 1:
                            connected_group.update(group)

                        # Now we look for valid opportunities to fill a single gap (one empty space) between chips
                        if len(group) == 1 and len(empty_spaces) == 1:
                            # We have one chip and one empty space between chips
                            # We check if placing a new chip in the empty space would create a connected group
                            (er, ec) = empty_spaces.pop()  # Get the empty space
                            '''if is_viable_group(group, [dr, dc]):  # Check if this potential group will form a group of 4
                                connected_group.add((er, ec))
                            else:
                                return None  # No valid group after this move'''
                            if not is_viable_group(group, [dr, dc]):  # Check if this potential group will form a group of 4
                                return None  # No valid group after this move
                    
                    '''if len(connected_group) >= 2:'''
                    if len(connected_group) >= 1:
                        connected_group_with_dir = [connected_group, [dr, dc]]
                        connected_groups.append(connected_group_with_dir)  # add the found connected group

                if connected_groups:
                    return connected_groups
                else:
                    return None  # No valid group found

            def find_best_move_for_group(group_with_dir):
                """
                Find the best move for expanding or infilling the largest connected group.
                The function checks for gaps that can be infilled, or places a chip in a direction
                that expands the group to potentially form a line of 4.
                """
                if group_with_dir is None:
                    # If no group can be infilled or expanded, fall back to random selection 
                    # Or available-row-based selection.
                    # Want to choose randomly between placing a random chip
                    # Or placing a chip in the column(s) with the most rows available.
                    options = [1, 2]
                    chosen_option = random.choice(options)
                    best_columns = []

                    if chosen_option == 1:
                        # choose a random column
                        for i in range(COLS):
                            best_columns.append(i)
                    else:
                        # choose the column with the most rows available
                        available_moves = self.get_legal_moves()
                        if not available_moves:
                            return None  # No moves available.

                        # Find columns with the most available rows.
                        max_rows_available = 0
                        for col in available_moves:
                            rows_available = sum(1 for row in range(ROWS) if self.board[row][col] == EMPTY)
                            if rows_available > max_rows_available:
                                max_rows_available = rows_available
                                best_columns = [col]
                            elif rows_available == max_rows_available:
                                best_columns.append(col)
                    
                    # Pick randomly (or randomly from the columns with the most available rows).
                    return random.choice(best_columns)

                else: 
                    best_move_for_group = None
                    largest_group_size = 0
                    group = group_with_dir[0]
                    direction = group_with_dir[1]

                    # If the group is size 1, iterate through the directions to find the best move that infills or expands the group
                    if len(group) == 1:
                        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                            for r, c in group:
                                # Try expanding in both directions (left-right, up-down, etc.)
                                for step in [-1, 1]:
                                    nr, nc = r, c
                                    for i in range(0, 4):  # Expand 3 steps in this direction including current space
                                        nr += dr * step
                                        nc += dc * step
                                        
                                        # Check if it's a valid move
                                        if 0 <= nr < ROWS and 0 <= nc < COLS:
                                            if self.board[nr][nc] == EMPTY:
                                                # Check if placing here would form a valid group of 4
                                                group_copy = group.copy()
                                                group_copy.add((nr, nc))
                                                if is_viable_group(group_copy, direction):
                                                    # If it's a viable group, consider this as a potential best move
                                                    if len(group_copy) > largest_group_size:
                                                        largest_group_size = len(group_copy)
                                                        best_move_for_group = nc  # Set the column as the best move
                    
                    # Otherwise, expand the group in its direction
                    else:
                        dr = direction[0]
                        dc = direction[1]
                        for r, c in group:
                            # Try expanding in both directions (left-right, up-down, etc.)
                            for step in [-1, 1]:
                                nr, nc = r, c
                                for i in range(0, 4):  # Expand 3 steps in this direction including current space
                                    nr += dr * step
                                    nc += dc * step
                                    
                                    # Check if it's a valid move
                                    if 0 <= nr < ROWS and 0 <= nc < COLS:
                                        if self.board[nr][nc] == EMPTY:
                                            # Check if placing here would form a valid group of 4
                                            group_copy = group.copy()
                                            group_copy.add((nr, nc))
                                            if is_viable_group(group_copy, direction):
                                                # If it's a viable group, consider this as a potential best move
                                                if len(group_copy) > largest_group_size:
                                                    largest_group_size = len(group_copy)
                                                    best_move_for_group = nc  # Set the column as the best move
                    

                    return best_move_for_group

            if block_opponent_win() is None:
                best_moves = []
                largest_group_size = 0
            
                # Check all possible groups of connected chips on the board.
                # For each group, check if expanding it would be viable.
                for col in range(COLS):
                    for row in range(ROWS):
                        if self.board[row][col] == COMPUTER:
                            # Look for a group of connected chips that can be expanded
                            groups = find_connected_groups(row, col)  # A helper function to find the group
                            
                            if groups:
                                for group_with_dir in groups:
                                    group = group_with_dir[0]
                                    best_move_for_group = find_best_move_for_group(group_with_dir)

                                    if best_move_for_group is not None:

                                        added_row = -1

                                        # getting the row and col of the best_move_for_group
                                        for r in range(ROWS):
                                            if self.board[r][best_move_for_group] == EMPTY:
                                                added_row = r
                                                break

                                        group_copy = group.copy()
                                        group_copy.add((added_row, best_move_for_group))

                                        if len(group_copy) > largest_group_size:
                                            largest_group_size = len(group_copy)
                                            best_moves = [best_move_for_group]  # Set the column as the best move
                                        elif len(group_copy) == largest_group_size:
                                            best_moves.append(best_move_for_group)
                if not best_moves:
                    return find_best_move_for_group(None) # no chips placed yet
                
                else:
                    # In the event of a tie, pick a random best move from the options
                    return random.choice(best_moves)
            else: 
                #print("blocked")
                return block_opponent_win()

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

            if self.is_full():
                self.print_board()
                print("It's a draw!")
                break

            self.print_board()

            if self.current_player == PLAYER:
                col = int(input("Choose a column (0-6): "))
            else:
                col = self.computer_move()
                print(f"Computer chooses column: {col}")
            
            if col is None:
                col = random.choice(self.get_legal_moves())

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

class PVP_ConnectFour():
    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.current_player = random.choice([PLAYER_1, PLAYER_2]) # randomize starting player

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
            if self.is_full():
                self.print_board()
                print("It's a draw!")
                break

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


def main():
    test = int(str(input("\n If you like to run an MCTS test, enter 2 for simple heuristic, 3 for advanced heuristic, and 4 for random. \n Otherwise, enter 0. \n\n Test mode: ")))
    if test > 0 and test <= 4:
        MCTS_wins = 0
        heuristic_wins = 0
        draws = 0
        for i in range(1000):
            game = CVC_ConnectFour(1, test, True)
            winner = game.play_game()
            print("Game number ", i + 1, ". Winner: player ", winner, " \n")
            if winner == "1":
                MCTS_wins += 1
            elif winner == "2":
                heuristic_wins += 1
            elif winner == "d":
                draws += 1
        
        heuristic = "MCTS"
        if test == 2:
            heuristic = "Simple"
        elif test == 3:
            heuristic = "Advanced"
        elif test == 4:
            heuristic = "Random"
        print(" MCTS won ", MCTS_wins, " out of 1000 games against ", heuristic.lower(), " heuristic. \n")
        print(" ", heuristic, " heuristic won ", heuristic_wins, " out of 1000 games against MCTS. \n")
        print(" There were ", draws, " draws out of 1000 games between MCTS and ", heuristic.lower(), " heuristic. \n")
    
    elif test == 5:
        adv_simple_wins = 0
        adv_rand_wins = 0
        simple_wins = 0
        random_wins = 0
        adv_simple_draws = 0
        adv_rand_draws = 0

        for i in range(1000):
            game = CVC_ConnectFour(3, 2, True)
            winner = game.play_game()
            print("Game number ", i + 1, ". Winner: player ", winner, " \n")
            if winner == "1":
                adv_simple_wins += 1
            elif winner == "2":
                simple_wins += 1
            elif winner == "d":
                adv_simple_draws += 1
        
        print(" Simple testing complete. \n")
        
        for i in range(1000):
            game = CVC_ConnectFour(3, 4, True)
            winner = game.play_game()
            print("Game number ", i + 1, ". Winner: player ", winner, " \n")
            if winner == "1":
                adv_rand_wins += 1
            elif winner == "2":
                random_wins += 1
            elif winner == "d":
                adv_rand_draws += 1
        
        print(" Random testing complete. \n")
        
        print(" Advanced heuristic won ", adv_simple_wins, " out of 1000 games against simple heuristic. \n")
        print(" Simple heuristic won ", simple_wins, " out of 1000 games against advanced heuristic. \n")
        print(" There were ", adv_simple_draws, " draws out of 1000 games between advanced heuristic and simple heuristic. \n")
        
        print(" Advanced heuristic won ", adv_rand_wins, " out of 1000 games against random heuristic. \n")
        print(" Random heuristic won ", random_wins, " out of 1000 games against advanced heuristic. \n")
        print(" There were ", adv_simple_draws, " draws out of 1000 games between advanced heuristic and random heuristic. \n")
    
    elif test == 6:
        test_modes = [4, 2, 3, 1]

        for test_mode in test_modes:
            
            MCTS_win_num_moves = {}
            heuristic_win_num_moves = {}
            
            for i in range(1000):
                game = CVC_ConnectFour(1, test_mode, True)
                winner = game.play_game()

                if i % 10 == 0:
                    print("Game number ", i + 1, ". Winner: player ", winner, " \n")
                
                if winner == "1":
                    num_moves = game.num_moves_computer_1
                    if num_moves in MCTS_win_num_moves:
                        MCTS_win_num_moves[num_moves] += 1
                    else:
                        MCTS_win_num_moves[num_moves] = 1

                elif winner == "2":
                    num_moves = game.num_moves_computer_2
                    if num_moves in heuristic_win_num_moves:
                        heuristic_win_num_moves[num_moves] += 1
                    else:
                        heuristic_win_num_moves[num_moves] = 1
            
            if test_mode == 1:
                plt.figure(figsize=(8, 8))
                plt.bar(list(MCTS_win_num_moves.keys()), MCTS_win_num_moves.values(), color='purple', width=0.7)
                plt.title('Distribution of Number of Moves for MCTS (1) win game vs. MCTS (2)')
                plt.xlabel('Number of Moves to Win')
                plt.ylabel('Number of Occurances of Win in Given Number of Moves')
                plt.savefig('MCTS_MCTS_1.png')
                plt.show()
                #plt.clf()

                plt.figure(figsize=(8, 8))
                plt.bar(list(heuristic_win_num_moves.keys()), heuristic_win_num_moves.values(), color='blue', width=0.7)
                plt.title('Distribution of Number of Moves for MCTS (2) win game vs. MCTS (1)')
                plt.xlabel('Number of Moves to Win')
                plt.ylabel('Number of Occurances of Win in Given Number of Moves')
                plt.savefig('MCTS_MCTS_2.png')
                plt.show()
                #plt.clf()

            elif test_mode == 2:
                plt.figure(figsize=(8, 8))
                plt.bar(list(MCTS_win_num_moves.keys()), MCTS_win_num_moves.values(), color='purple', width=0.7)
                plt.title('Distribution of Number of Moves for MCTS win game vs. Simple Heuristic')
                plt.xlabel('Number of Moves to Win')
                plt.ylabel('Number of Occurances of Win in Given Number of Moves')
                plt.savefig('MCTS_simple_1.png')
                plt.show()
                #plt.clf()

                plt.figure(figsize=(8, 8))
                plt.bar(list(heuristic_win_num_moves.keys()), heuristic_win_num_moves.values(), color='blue', width=0.7)
                plt.title('Distribution of Number of Moves for Simple Heuristic win game vs. MCTS')
                plt.xlabel('Number of Moves to Win')
                plt.ylabel('Number of Occurances of Win in Given Number of Moves')
                plt.savefig('MCTS_simple_2.png')
                plt.show()
                #plt.clf()

            elif test_mode == 3:
                plt.figure(figsize=(8, 8))
                plt.bar(list(MCTS_win_num_moves.keys()), MCTS_win_num_moves.values(), color='purple', width=0.7)
                plt.title('Distribution of Number of Moves for MCTS win game vs. Advanced Heuristic')
                plt.xlabel('Number of Moves to Win')
                plt.ylabel('Number of Occurances of Win in Given Number of Moves')
                plt.savefig('MCTS_adv_1.png')
                plt.show()
                #plt.clf()

                plt.figure(figsize=(8, 8))
                plt.bar(list(heuristic_win_num_moves.keys()), heuristic_win_num_moves.values(), color='blue', width=0.7)
                plt.title('Distribution of Number of Moves for Advanced Heuristic win game vs. MCTS')
                plt.xlabel('Number of Moves to Win')
                plt.ylabel('Number of Occurances of Win in Given Number of Moves')
                plt.savefig('MCTS_adv_2.png')
                plt.show()
                #plt.clf()
    
            elif test_mode == 4:
                plt.figure(figsize=(8, 8))
                plt.bar(list(MCTS_win_num_moves.keys()), MCTS_win_num_moves.values(), color='purple', width=0.7)
                plt.title('Distribution of Number of Moves for MCTS win game vs. Random Heuristic')
                plt.xlabel('Number of Moves to Win')
                plt.ylabel('Number of Occurances of Win in Given Number of Moves')
                plt.savefig('MCTS_rand_1.png')
                plt.show()
                #plt.clf()

                plt.figure(figsize=(8, 8))
                plt.bar(list(heuristic_win_num_moves.keys()), heuristic_win_num_moves.values(), color='blue', width=0.7)
                plt.title('Distribution of Number of Moves for Random Heuristic win game vs. MCTS')
                plt.xlabel('Number of Moves to Win')
                plt.ylabel('Number of Occurances of Win in Given Number of Moves')
                plt.savefig('MCTS_rand_2.png')
                plt.show()
                #plt.clf()
    
    elif test == 0:
        mode = str(input("\n 1) Enter 'cvc' or 1 to run a computer vs. computer game \n 2) Enter 'pvc' or 2 to run a player vs. computer game \n 3) Enter 'pvp' or 3 to run a player vs. player game \n\n Input: "))
        if mode == "cvc" or mode == "1":
            policy_1 = int(input("\n\n What strategy / policy would you like computer 1 to use? \n Enter the number corresponding to the desired strategy: \n 1) Monte Carlo Tree Search \n 2) Simple Heuristic \n 3) Adanvanced Heuristic \n 4) Random \n\n Input: "))
            policy_2 = int(input("\n\n What strategy / policy would you like compueter 2 to use? \n Enter the number corresponding to the desired strategy: \n 1) Monte Carlo Tree Search \n 2) Simple Heuristic \n 3) Adanvanced Heuristic \n 4) Random \n\n Input: "))
            game = CVC_ConnectFour(policy_1, policy_2, False)
        elif mode == "pvc" or mode == "2":
            policy = int(input("\n\nWhat strategy / policy would you like the computer to use? \n Enter the number corresponding to the desired strategy: \n 1) Monte Carlo Tree Search \n 2) Simple Heuristic \n 3) Adanvanced Heuristic \n 4) Random \n\n Input: "))
            game = PVC_ConnectFour(policy)
        elif mode == "pvp" or mode == "3":
            game = PVP_ConnectFour()
        else:
            raise Exception("Error: please enter a valid game mode")
        
        print("\n\n")
        game.play_game()
        #result = game.play_game()
        #print("player ", result, " wins!")

    else:
            raise Exception("Error: please enter a valid game mode")

if __name__ == "__main__":
    main()

# COMMENT OUT LINES 116 AND 104 IN MCTS.PY