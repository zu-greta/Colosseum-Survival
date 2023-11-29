# copy - minimax + ideas
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
        self.max_depth = 3

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
        _, action = self.minimax(chess_board, my_pos, adv_pos, max_step, self.max_depth, float('-inf'), float('inf'), True)
        time_taken = time.time() - start_time
        
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, self.dir_map["u"]
    
    def evaluate_board(self, chess_board, my_pos, adv_pos, max_step):
        # Add your own evaluation function based on the game state
        # The higher the score, the better the position for the bot
        return 0

    def is_terminal_node(self, depth):
        # Add your own conditions to check if it's a terminal node
        return depth == 0

    def minimax(self, chess_board, my_pos, adv_pos, max_step, depth, alpha, beta, maximizing_player):
        if self.is_terminal_node(depth):
            return self.evaluate_board(chess_board, my_pos, adv_pos, max_step), None

        valid_actions = ["u", "r", "d", "l"]

        if maximizing_player:
            max_eval = float('-inf')
            best_action = None
            for action in valid_actions:
                new_pos = self.get_new_position(my_pos, action)
                new_board = self.simulate_move(chess_board, my_pos, new_pos, action)
                eval, _ = self.minimax(new_board, new_pos, adv_pos, max_step, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_action = action
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_action
        else:
            min_eval = float('inf')
            best_action = None
            for action in valid_actions:
                new_pos = self.get_new_position(adv_pos, action)
                new_board = self.simulate_move(chess_board, adv_pos, new_pos, action)
                eval, _ = self.minimax(new_board, my_pos, new_pos, max_step, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_action = action
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_action

    def get_new_position(self, pos, action):
        x, y = pos
        if action == "u":
            return x - 1, y
        elif action == "r":
            return x, y + 1
        elif action == "d":
            return x + 1, y
        elif action == "l":
            return x, y - 1

    def simulate_move(self, chess_board, start_pos, end_pos, action):
        # Add your own logic to simulate the move on the chess board
        # This may include updating the positions of both the bot and the opponent
        return chess_board.copy()
    

"""
# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time

from collections import defaultdict




@register_agent("student_agent")
class StudentAgent(Agent):
    
    #A dummy class for your implementation. Feel free to use this class to
    #add any helper functionalities needed for your agent.
    

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    class MCTS():
        def __init__(self, state, parent=None, parent_action=None):
            self.state = state
            self.parent = parent
            self.parent_action = parent_action
            self.children = []
            self._number_of_visits = 0
            self._results = defaultdict(int)
            self._results[1] = 0
            self._results[-1] = 0
            self._untried_actions = None
            self._untried_actions = self.untried_actions()
            return
            # (self.T / self.N) + c * np.sqrt(np.log(top_node.N) / self.N) 
        
        def untried_actions(self):
            self._untried_actions = self.get_legal_actions()
            return self._untried_actions
        def q(self):
            wins = self._results[1]
            loses = self._results[-1]
            return wins - loses
        def n(self):
            return self._number_of_visits
        def expand(self):
            action = self._untried_actions.pop()
            next_state = self.move(action)
            child_node = MCTS(next_state, parent=self, parent_action=action)
            self.children.append(child_node)
            return child_node 
        def is_terminal_node(self):
            return self.is_game_over()
        def rollout(self):
            current_rollout_state = self
            while not current_rollout_state.is_game_over():  
                possible_moves = current_rollout_state.get_legal_actions()         
                action = self.rollout_policy(possible_moves)
                current_rollout_state = current_rollout_state.move(action)
            return current_rollout_state.game_result()
        def backpropagate(self, result):
            self._number_of_visits += 1.
            self._results[result] += 1.
            if self.parent:
                self.backpropagate(result)
        def is_fully_expanded(self):
            return len(self._untried_actions) == 0
        def best_child(self, c_param=0.1):
            choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
            return self.children[np.argmax(choices_weights)]
        def rollout_policy(self, possible_moves):
            return possible_moves[np.random.randint(len(possible_moves))]
        def _tree_policy(self):
            current_node = self
            while not current_node.is_terminal_node():
                if not current_node.is_fully_expanded():
                    return current_node.expand()
                else:
                    current_node = current_node.best_child()
            return current_node
        def best_action(self):
            simulation_no = 100
            for i in range(simulation_no):
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
            return self.best_child(c_param=0.)
        def get_legal_actions(self): 
            x_max, y_max, _ = chess_board.shape
            legal_actions = []
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            # Enumerate through all possible directions and positions to place walls
            for direction in directions:
                new_x = my_pos[0] + direction[0]
                new_y = my_pos[1] + direction[1]
                # Check if the new position is within the board boundaries
                if 0 <= new_x < x_max and 0 <= new_y < y_max:
                    # Check if the new position is not a wall and not occupied by the adversary
                    if not chess_board[new_x, new_y, 0] and (new_x, new_y) != adv_pos:
                        # Check if the new position is not setting a boundary on a boundary
                        if not (chess_board[new_x, new_y, 1] and chess_board[my_pos[0], my_pos[1], 1]):
                            # Check if placing a wall is within the max_step constraint
                            if max_step > 0:
                                legal_actions.append(((new_x, new_y), direction))
            return legal_actions
        def is_game_over(self):
            x_max, y_max, _ = chess_board.shape
            moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
            self.p0_pos = np.random.randint(0, x_max, size=2)
            self.p1_pos = y_max - 1 - self.p0_pos
            # Union-Find
            father = dict()
            for r in range(x_max):
                for c in range(y_max):
                    father[(r, c)] = (r, c)
            def find(pos):
                if father[pos] != pos:
                    father[pos] = find(father[pos])
                return father[pos]
            def union(pos1, pos2):
                father[pos1] = pos2
            for r in range(x_max):
                for c in range(y_max):
                    for dir, move in enumerate(
                        moves[1:3]
                    ):  # Only check down and right
                        if chess_board[r, c, dir + 1]:
                            continue
                        pos_a = find((r, c))
                        pos_b = find((r + move[0], c + move[1]))
                        if pos_a != pos_b:
                            union(pos_a, pos_b)
            for r in range(x_max):
                for c in range(y_max):
                    find((r, c))
            p0_r = find(tuple(self.p0_pos))
            p1_r = find(tuple(self.p1_pos))
            if p0_r == p1_r:
                return False
            return True
        def game_result(self):
            x_max, y_max, _ = chess_board.shape
            moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
            self.p0_pos = np.random.randint(0, x_max, size=2)
            self.p1_pos = y_max - 1 - self.p0_pos
            # Union-Find
            father = dict()
            for r in range(x_max):
                for c in range(y_max):
                    father[(r, c)] = (r, c)
            def find(pos):
                if father[pos] != pos:
                    father[pos] = find(father[pos])
                return father[pos]
            def union(pos1, pos2):
                father[pos1] = pos2
            for r in range(x_max):
                for c in range(y_max):
                    for dir, move in enumerate(
                        moves[1:3]
                    ):  # Only check down and right
                        if chess_board[r, c, dir + 1]:
                            continue
                        pos_a = find((r, c))
                        pos_b = find((r + move[0], c + move[1]))
                        if pos_a != pos_b:
                            union(pos_a, pos_b)
            for r in range(x_max):
                for c in range(y_max):
                    find((r, c))
            p0_r = find(tuple(self.p0_pos))
            p1_r = find(tuple(self.p1_pos))
            p0_score = list(father.values()).count(p0_r)
            p1_score = list(father.values()).count(p1_r)
            
            player_win = None
            if p0_score > p1_score:
                player_win = 1
            elif p0_score < p1_score:
                player_win = 0
            else:
                player_win = -1  # Tie
            return player_win
            
        def move(self,action):
            # use action to decide what move to make
            # update the chess board wall and new agent position
            (x, y), dir = action
            chess_board[x, y, dir] = True
            return chess_board

    class MCTS():
        def __init__(self):
            pass
        def getUCBscore(self):
            c = 0.1
            # Unexplored nodes have maximum values so we favour exploration
            if self.N == 0:
                return float('inf')
            # We need the parent node of the current node 
            top_node = self
            if top_node.parent:
                top_node = top_node.parent
            # MCTS formula for calculating the node value
            return (self.T / self.N) + c * np.sqrt(np.log(top_node.N) / self.N) 
            # T - score, N - visits, c - exploration parameter, top_node.N - visits of parent node
        def selection(self):    
            pass    
        def expansion(self):    
            pass
        def simulation(self):
            pass
        def backpropagation(self):  
            pass  

    def step(self, chess_board, my_pos, adv_pos, max_step):
        
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
        

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        time_taken = time.time() - start_time     

        

        root = self.MCTS(state = chess_board)
        selected_node = root.best_action()
        my_pos, dir = selected_node.parent_action

        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, dir #self.dir_map["u"]
"""