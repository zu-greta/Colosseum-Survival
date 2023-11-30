# win rate: 0.7 against student_agent with 10 games and 1.905 time [not random sim]
# win rate: 0.9 against student_agent with 10 games and 1.905 time [not random sim]
# win rate: ? against random_agent with 100 games and ? time [not random sim]

# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time

import json

@register_agent("student14_agent")
class Student14Agent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(Student14Agent, self).__init__()
        self.name = "Student14Agent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.max_time = 1.9 # max time for each step
        self.max_sims = 5 # max simulations for each step
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1)) # up, right, down, left

    # check if time out
    def timeout(self, start_time):
        return time.time() - start_time > self.max_time
    
    # get all legal moves in order of priority based on heuristic function
    def all_moves(self, chess_board, my_pos, adv_pos, max_step, start_time):
        # return a list of legal moves using BFS
        # for every possible num of steps until max_step, travel 4 directions + put wall. append to legal, sort and return the top #
        legal = [] # list of [p, s, n, (x, y), dir]
        state_queue = [(my_pos, 0)]
        visited = [my_pos, adv_pos]
        # BFS
        while state_queue and not self.timeout(start_time):
            cur_pos, cur_step = state_queue.pop()
            x, y = cur_pos
            dis = self.calculate_distance(cur_pos, adv_pos) #get the distance between my current position and adv
            for dir, move in enumerate(self.moves): # 4 directions
                if chess_board[x, y, dir]: # if there is a wall, skip the move
                    continue
                #check if gameover, me winner = 1, adv winner = -1, tie = 0
                self.set_barrier(x, y, dir, chess_board, True)
                over, w = self.is_gameover((x, y), adv_pos, chess_board)
                self.set_barrier(x, y, dir, chess_board, False)
                if over and w: # gameover, get winner and set p as the returned winner
                    p = w 
                    if p == 1:
                        return [[1, 1, 1, (x, y), dir]] #immediate win, just play that move
                else:
                    # Defense heuristic
                    if sum([chess_board[x, y, i] for i in range(4)]) >= 2: # if there are 2 walls, p = 0.1 
                        walls = 0.1
                    else: 
                        walls = 1
                    # Heuristic function: offense distance * defense walls * offense direction
                    p = (1 - dis/20) * walls * self.calculate_direction((x,y), adv_pos, dir)
                legal.append([p, 0, 1, (x, y), dir]) # append to legal
                # get the next step
                new_x, new_y = x + move[0], y + move[1] 
                new_pos = (new_x, new_y)
                if new_pos not in visited and cur_step + 1 <= max_step: # if not visited and not exceed max_step, append to queue
                    visited.append(new_pos)
                    state_queue.append((new_pos, cur_step + 1))
        return sorted(legal, key=lambda x: x[0], reverse=True)[:10] # sort and return top #
    
    # calculate the manhattan distance between two positions
    def calculate_distance(self, my_pos, adv_pos):
        r0, c0 = my_pos
        r1, c1 = adv_pos
        return abs(r0 - r1) + abs(c0 - c1)
    
    # calculate the direction of the adversary (for offensive heuristic)
    def calculate_direction(self, my_pos, adv_pos, dir):
        r0, c0 = my_pos
        r1, c1 = adv_pos
        if (r0 <= r1 and dir == 2) or (r0 >= r1 and dir == 0) or (c0 <= c1 and dir == 1) or (c0 >= c1 and dir == 3):
            return 1
        else:
            return 0
    
    # build or remove barrier
    def set_barrier(self, r, c, dir, step_board, is_set):
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        # Set the barrier to True/False
        step_board[r, c, dir] = is_set
        # Set the opposite barrier to True/False
        move = self.moves[dir]
        step_board[r + move[0], c + move[1], opposites[dir]] = is_set
    
    # randomly select from the best moves simulation
    def simulation(self, my_pos, dir, adv_pos, max_step, step_board, start_time):
        # take the current step
        self.set_barrier(my_pos[0], my_pos[1], dir, step_board, True)
        player_switch = -1 # -1 is adv, 1 is me. every time switch player do player_switch *= -1
        score = 0 # score = 1 if win, score = -1 if lose, score = 0 if tie
        gameover = False
        while (not gameover and not self.timeout(start_time)) :
            # is_gamover is true if one of the player is enclosed, return the winner
            if (player_switch == -1) :
                # adversary's turn
                a_children = self.all_moves(step_board, adv_pos, my_pos, max_step, start_time)
                if len(a_children) == 0:
                    gameover, score = self.is_gameover(my_pos, adv_pos, step_board)
                    break
                else:
                    n = np.random.randint(0, min(5, len(a_children)))
                    adv_pos, dir = a_children[n][3], a_children[n][4]
                    self.set_barrier(adv_pos[0], adv_pos[1], dir, step_board, True)
            else :
                # my turn
                m_children = self.all_moves(step_board, my_pos, adv_pos, max_step, start_time)
                if len(m_children) == 0:
                    gameover, score = self.is_gameover(my_pos, adv_pos, step_board)
                    break
                else:
                    n = np.random.randint(0, min(5, len(m_children)))
                    my_pos, dir = m_children[n][3], m_children[n][4]
                    self.set_barrier(my_pos[0], my_pos[1], dir, step_board, True)
            gameover, score = self.is_gameover(my_pos, adv_pos, step_board) # check if gameover
            player_switch *= -1 # switch player
        # run random simulation on x, y, d node. if win, s+=1, lose, s-=1. tie s+=0
        return score
    
    # check if gameover
    def is_gameover(self, my_pos, adv_pos, step_board):
        # return list of spaces reachable from pos
        def find(pos):
            visited = [pos]
            path_queue = [pos]
            while path_queue:
                cur_pos = path_queue.pop()
                x, y = cur_pos
                for dir, move in enumerate(self.moves):
                    if not step_board[x, y, dir]:
                        new_x, new_y = x + move[0], y + move[1]
                        new_pos = (new_x, new_y)
                        if new_pos not in visited:
                            visited.append(new_pos)
                            path_queue.append(new_pos)
            return visited
        find_my_pos = find(my_pos) # list of spaces reachable from my_pos
        if adv_pos in find_my_pos: # if adv_pos is in find_my_pos the game is not over
            return False, 0
        # game is over, check if I win or lose
        find_adv_pos = find(adv_pos) # list of spaces reachable from adv_pos
        if len(find_adv_pos) == len(find_my_pos): # if the two lists are the same, it's a tie
            return True, 0
        elif len(find_adv_pos) > len(find_my_pos): # if find_adv_pos is larger, I lose
            return True, -1
        else: # if find_my_pos is larger, I win
            return True, 1
        
    # get the best next move
    def best_move(self, children):
        # return a tuple of ((x, y), dir),
        # where (x, y) is the next position of your agent and dir is the direction of the wall you want to put on.
        # [(p, s, n, (x, y), dir), ...]
        best_next_move = sorted(children, key=lambda x: ((x[1]/x[2], x[0])), reverse=True)[0]
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
        chess_board_list = chess_board.tolist() # convert chess_board to list -> json format
        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        # get all legal moves in order of priority based on heuristic function
        # [(p, s, n, (x, y), dir), ...] -> [0: p, 1: s, 2: n, 3: (x, y), 4: dir]
        children = self.all_moves(chess_board, my_pos, adv_pos, max_step, start_time) # get top legal moves
        if (len(children) == 1): # only one move (usually immediate win)
            my_pos, dir = children[0][3], children[0][4]
        elif (len(children) != 0): # more than one move, run simulations to find the best move
            for i in range(len(children)): # run simulations on each child
                while (children[i][2] < self.max_sims and not self.timeout(start_time)): # run until max_sims or timeout
                    #step_board = deepcopy(chess_board) # copy the chess_board
                    step_board = np.array(json.loads(json.dumps(chess_board_list))) # copy the chess_board
                    score = self.simulation(children[i][3], children[i][4], adv_pos, max_step, step_board, start_time) # run simulation
                    children[i][1] += score # update score
                    children[i][2] += 1 # update num of simulations
                #print("num:", children[i])
                if (self.timeout(start_time)): # if timeout, break
                    break
            my_pos, dir = self.best_move(children) # get the best move based on score/num_sims

        time_taken = time.time() - start_time     
        print("My AI's turn took ", time_taken, "seconds.")
        return my_pos, dir 
