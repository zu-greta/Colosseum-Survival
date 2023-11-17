# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
def is_surrounded(self, chess_board, pos):
    # Heuristic 1: Check for being surrounded by 3 walls after your turn  
        r, c = pos
        walls_count = sum(chess_board[r, c, :])
        return walls_count == 3
    
    def get_last_free_wall(self, chess_board, pos):
    # returns the 4th wall if 3 others are drawn
        r, c = pos
        if chess_board[r, c, 0] == False:
            return chess_board[r, c, 0]
        elif chess_board[r, c, 1] == False:
            return chess_board[r, c, 1]
        elif chess_board[r, c, 2] == False:
            return chess_board[r, c, 2]
        elif chess_board[r, c, 3] == False:
            return chess_board[r, c, 3]
        
    def count_move_options(self, chess_board, pos):
    # Heuristic 2: Check for how many move options you would have after ending
        moves = [0, 1, 2, 3]
        r, c = pos
        move_options = sum(1 for move in moves if not chess_board[r, c, moves.index(move)])
        return move_options

    
    def calculate_distance(self, my_pos, adv_pos):
    # Heuristic 3: Check your distance from the opponent
        return abs(my_pos[0] - adv_pos[0]) + abs(my_pos[1] - adv_pos[1])

    
    def count_walls(self, chess_board):
    # Heuristic 4: Overall number of walls on the board
        return np.sum(chess_board)




def is_valid_wall_placement(self, chess_board, pos, direction):
    # Check if placing a wall at the specified position and direction is valid
        return chess_board[pos[0], pos[1], direction]

    
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
        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, self.dir_map["u"]
