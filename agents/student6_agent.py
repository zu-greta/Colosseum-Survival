# from quinn

# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from collections import deque


@register_agent("student6_agent")
class Student6Agent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(Student6Agent, self).__init__()
        self.name = "Student6Agent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

        # Moves (Up, Right, Down, Left)
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Initialize aggressive proximity count
        self.aggressive_proximity_count = 0

    # Heuristic 1: Check for being surrounded by 3 walls after your turn
    def is_surrounded(self, chess_board, pos):
        r, c = pos
        walls_count = sum(chess_board[r, c, :])
        return walls_count == 3


    # Helper bool method to check if a move is inside the board
    def check_boundary(self, chess_board, pos):
        r, c = pos
        return 0 <= r < chess_board[0] and 0 <= c < chess_board[1]

    # Heuristic 2: Check for how many move options you would have after ending / Mobility
    def count_move_options(self, chess_board, pos):

        move_options = 0  # Initialize the count of move options
        visited = set()  # Set to keep track of visited positions
        queue = deque([(pos, 0)])  # Initialize the BFS queue with the starting position and step count

        while queue:
            current_pos, current_steps = queue.popleft()

            for direction in range(4):
                # Check if the move in the current direction is valid (not blocked by a wall)
                if not chess_board[current_pos[0], current_pos[1], direction]:
                    # Calculate the next position based on the current direction
                    next_pos = (
                        current_pos[0] + self.moves[direction][0],
                        current_pos[1] + self.moves[direction][1],
                    )

                    # Check if the next position is within the boundaries and not visited
                    if self.check_boundary(chess_board, next_pos) and next_pos not in visited:
                        visited.add(next_pos)  # Mark the position as visited
                        queue.append((next_pos, current_steps + 1))  # Add the next position to the queue

                        # Increment move_options for each valid move
                        move_options += 1

        return move_options

    # Heuristic 3: Check your distance from the opponent
    def calculate_distance(self, my_pos, adv_pos):
        r0, c0 = my_pos
        r1, c1 = adv_pos
        return abs(r0 - r1) + abs(c0 - c1)

    # Heuristic 4: Overall number of walls on the board
    def count_walls(self, chess_board):
        return np.sum(chess_board)

    # Heuristic 5: "Continuing" a wall or pattern    
    def is_pattern(self, chess_board, my_pos):
        pass

    # Heuristic 6: Trapping enemy
    def is_enemy_trappable(self, chess_board, my_pos, adv_pos, max_step):
         if self.is_surrounded(chess_board, adv_pos) and self.calculate_distance( my_pos, adv_pos) <= max_step:
             return True
         else:
             return False

    # Evaluate the heuristic if aggressive gameplay is chosen
    def evaluate_aggressive_heuristic(self, chess_board, my_pos, adv_pos, max_step):

        if self.is_surrounded( chess_board, my_pos):
            # Avoid being surrounded
            score_surrounded = -20
        else:
            score_surrounded = 0

        if self.is_enemy_trappable(chess_board, my_pos, adv_pos, max_step):
            # If the enemy can be trap, it's a guaranteed win
            score_trap = 100
        else:
            score_trap = 0

        score_move_options = self.count_move_options(chess_board, my_pos)   # Positive value: to go for move that give you mobility
        score_proximity = self.calculate_distance(my_pos, adv_pos)          # Positive value: to close to enemy for aggressive gameplay

        total_score = score_surrounded + score_move_options + score_proximity + score_trap
        return total_score

    # Evaluate the heuristic if passive gameplay is chosen
    def evaluate_passive_heuristic(self, chess_board, my_pos, adv_pos, max_step):

        if self.is_surrounded( chess_board, my_pos):
            # Avoid being surrounded
            score_surrounded = -20
        else:
            score_surrounded = 0

        if self.is_enemy_trappable(chess_board, my_pos, adv_pos, max_step):
            # If the enemy can be trap, it's a guaranteed win
            score_trap = 100
        else:
            score_trap = 0

        score_move_options = self.count_move_options(chess_board, my_pos)   # Positive value: to go for move that give you mobility
        score_proximity = - (self.calculate_distance(my_pos, adv_pos))      # Negative value: to avoid enemy for passive gameplay

        total_score = score_surrounded + score_move_options + score_proximity + score_trap
        return total_score

    # Increment aggressive_proximity_count if a distane is aggressive
    def is_in_proximity(self, chess_board, opponent_distance):
        """ 
            6x6, 7x7 -> 1 dist. is considered aggressive
            8x8, 9x9, 10x10 -> 2 dist. is considered aggressive
            11x11, 12x12 -> 3 dist. is considered aggressive 

        """
        if chess_board[0] == 6 or chess_board[0] == 7:
            if opponent_distance >= 1:
                return True
        elif chess_board[0] == 8 or chess_board[0] == 9 or chess_board[0] == 10:
            if opponent_distance >= 2:
                return True
        elif chess_board[0] == 11 or chess_board[0] == 12:
            if opponent_distance >= 3:
                return True
        else:
            False
        
    #Checks if the aggressive_proximity_count met the condition to have the enemy be considered as aggressive
    def is_aggressive(self, chess_board, aggressive_proximity_count):
        return aggressive_proximity_count >= chess_board[0] // 3
    

    def evaluate_heuristic(self, chess_board, my_pos, adv_pos, max_step):
        # Implement your heuristic here based on the opponent's aggressiveness
        # Example: Use different heuristics for aggressive and passive opponents

        if self.is_in_proximity(chess_board, self.calculate_distance( my_pos, adv_pos)):
            self.aggressive_proximity_count += 1
        else:
            if self.aggressive_proximity_count > 1:
                self.aggressive_proximity_count -= 1

        if self.is_aggressive(chess_board, self.aggressive_proximity_count):
            return self.evaluate_aggressive_heuristic(chess_board, my_pos, adv_pos, max_step)
        else:
            return self.evaluate_passive_heuristic(chess_board, my_pos, adv_pos, max_step)



    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()

        opponent_distance = self.calculate_distance(my_pos, adv_pos)


        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, self.dir_map["u"]
