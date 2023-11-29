# mcts 2 but no class + engame check + heuristics?
# win rate 1.0

# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time

from collections import deque




@register_agent("student10_agent")
class Student10Agent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(Student10Agent, self).__init__()
        self.name = "Student10Agent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.max_time = 10.9 # max time per move
        self.m = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def timeout(self, start_time):
        return time.time() - start_time > self.max_time

    
    # get all legal moves
    def all_moves(self, chess_board, my_pos, adv_pos, max_step):
        # return a list of legal moves using BFS
        # for every possible num of steps until max_step, travel 4 directions + put wall. append to legal and return
        legal = []
        state_queue = [(my_pos, 0)]
        visited = [(my_pos)]
        #distance = self.calculate_distance(my_pos, adv_pos) # get distance between me and adv
        # BFS
        while state_queue:
            cur_pos, cur_step = state_queue.pop(0)
            x, y = cur_pos
            dis = self.calculate_distance(cur_pos, adv_pos) #get the distance between my current position and adv
            for dir, move in enumerate(self.m):
                if chess_board[x, y, dir]:
                    continue
                #check if gameover, winner = me, winner = adv, tie
                self.set_barrier(x, y, dir, chess_board)
                over, w = self.is_gameover((x, y), adv_pos, chess_board)
                self.remove_barrier(x, y, dir, chess_board)
                if over and w:
                    p = w 
                    if p == 1:
                        return [[1, 1, 1, (x, y), dir]]
                else:
                    if sum([chess_board[x, y, i] for i in range(4)]) >= 2: 
                        walls = 0 
                    else: 
                        walls = 1
                    p = (1 - dis/20) * walls * self.calculate_direction((x,y), adv_pos, dir)
                legal.append([p, 0, 1, (x, y), dir])
                new_x, new_y = x + move[0], y + move[1]
                new_pos = (new_x, new_y)
                if new_pos == adv_pos or new_pos in visited:
                    continue
                if cur_step + 1 <= max_step:
                    visited.append(new_pos)
                    state_queue.append((new_pos, cur_step + 1))
        return sorted(legal, key=lambda x: x[0], reverse=True)
    
    def calculate_distance(self, my_pos, adv_pos):
        r0, c0 = my_pos
        r1, c1 = adv_pos
        return abs(r0 - r1) + abs(c0 - c1)
    
    def calculate_direction(self, my_pos, adv_pos, dir):
        r0, c0 = my_pos
        r1, c1 = adv_pos
        if (r0 <= r1 and dir == 2) or (r0 >= r1 and dir == 0) or (c0 <= c1 and dir == 1) or (c0 >= c1 and dir == 3):
            return 1
        else:
            return 0
        
    def set_barrier(self, r, c, dir, step_board):
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        # Set the barrier to True
        step_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.m[dir]
        step_board[r + move[0], c + move[1], opposites[dir]] = True
    
    def remove_barrier(self, r, c, dir, step_board):
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        # Set the barrier to False
        step_board[r, c, dir] = False
        # Set the opposite barrier to False
        move = self.m[dir]
        step_board[r + move[0], c + move[1], opposites[dir]] = False

    # simulate game
    def simulate(self, my_pos, dir, adv_pos, max_step, step_board, start_time):
        # take the current step
        self.set_barrier(my_pos[0], my_pos[1], dir, step_board)
        player_switch = -1
        score = 0
        # player = adv
        # random walk
        # check if gameover
        # if gameover, return score
        # else, switch player and return to random walk step (loop)
        # check time somewhere in loop. if time out, break and return score
        gameover = False
        while (gameover == False) :
            # is_gamover is true if one of the player is enclosed. return bool
            if (player_switch == -1) :
                # adversary's turn
                a_children = self.all_moves(step_board, adv_pos, my_pos, max_step)
                if len(a_children) == 0:
                    gameover, score = self.is_gameover(my_pos, adv_pos, step_board)
                    break
                else:
                    n = np.random.randint(0, min(5, len(a_children)))
                    adv_pos, dir = a_children[n][3], a_children[n][4]
                    self.set_barrier(adv_pos[0], adv_pos[1], dir, step_board)
            else :
                # my turn
                children = self.all_moves(step_board, my_pos, adv_pos, max_step)
                if len(children) == 0:
                    gameover, score = self.is_gameover(my_pos, adv_pos, step_board)
                    break
                else:
                    n = np.random.randint(0, min(5, len(children)))
                    my_pos, dir = children[n][3], children[n][4]
                    self.set_barrier(my_pos[0], my_pos[1], dir, step_board)
            gameover, score = self.is_gameover(my_pos, adv_pos, step_board)
            player_switch *= -1
            if (self.timeout(start_time)):
                break
        # run random simulation on x, y, d node. if win, s+=1, lose, s-=1. tie s+=0
        return score
    
    def is_gameover(self, my_pos, adv_pos, step_board):
        x_max, y_max, _ = step_board.shape
        winner = 0
        # check if gameover, if yes, gameover = True
        # 1 is me, -1 is adv, 0 is tie
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
                for dir, move in enumerate(self.m[1:3]):
                    if step_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, winner
        if p0_score > p1_score:
            winner = 1
        elif p0_score < p1_score:
            winner = -1
        else:
            winner = 0
        return True, winner
    
    # get the best next move
    def best_move(self, children):
        # return a tuple of ((x, y), dir),
        # where (x, y) is the next position of your agent and dir is the direction of the wall you want to put on.
        best_next_move = sorted(children, key=lambda x: (x[1]/x[2]), reverse=True)[0]
        return best_next_move[3], best_next_move[4]
    
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

        # children = [] # list of children nodes with each child node = [0: priority, 1: score, 2: num_sims, 3: pos, 4: dir]
        # find all legal moves based on my_pos
        children = self.all_moves(chess_board, my_pos, adv_pos, max_step)
        if children[0][0] == 1 or children[0][0] == -1: #can win or immediate loss the game right away
            my_pos, dir = children[0][3], children[0][4]
        # else run simulations and choose best move
        else:
            for i in range(len(children)):
                if children[i][0] == -1:
                    continue # skip the bad move #don't continue looking at losing moves
                while (children[i][2] < 5 and not self.timeout(start_time)): # chess_board.shape[0]
                    step_board = deepcopy(chess_board)
                    score = self.simulate(children[i][3], children[i][4], adv_pos, max_step, step_board, start_time) 
                    children[i][1] += score
                    children[i][2] += 1
                #print("HI: ", children[i], "\n")
                if (self.timeout(start_time)):
                    break
            my_pos, dir = self.best_move(children)
        time_taken = time.time() - start_time
        print("My AI's turn took ", time_taken, "seconds.")
        return my_pos, dir
