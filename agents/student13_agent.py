# 0.98, 1.91664 max time, 20 max sims
# win 0.992, maxi-time 2.05s, 20 max sims, with 1000 games
# win 1.0, maxi-time 1.913s, 20 max sims, with 100 games [new gameover]
# win 1.0, maxi-time 1.903s, 20 max sims, with 100 games [max 20 moves]
# win 1.0, maxi-time 1.939s, 20 max sims, with 1000 games [max 20 moves, 0.1 walls]

# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time

@register_agent("student13_agent")
class Student13Agent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(Student13Agent, self).__init__()
        self.name = "Student13Agent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.max_time = 1.9
        self.max_sims = 20
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

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
            for dir, move in enumerate(self.moves):
                if chess_board[x, y, dir]:
                    continue
                #check if gameover, winner = me, winner = adv, tie
                self.set_barrier(x, y, dir, chess_board)
                over, w = self.is_gameover((x, y), adv_pos, chess_board)
                self.remove_barrier(x, y, dir, chess_board)
                if over and w:
                    p = w 
                    #print("x, y, dir, w: ", x, y, dir, w)
                    if p == 1:
                        return [[1, 1, 1, (x, y), dir]]
                else:
                    if sum([chess_board[x, y, i] for i in range(4)]) >= 2: 
                        walls = 0.1
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
        return sorted(legal, key=lambda x: x[0], reverse=True)[:20]
    
    def calculate_distance(self, my_pos, adv_pos):
        r0, c0 = my_pos
        r1, c1 = adv_pos
        return abs(r0 - r1) + abs(c0 - c1)
    
    # get offensive moves
    def offensive_moves(self, chess_board, my_pos, adv_pos, max_step):
        # find a path to closest position near adv_pos and store it in a list
        # store in visited
        # using BFS, backtrack from adv_pos to my_pos
        # [[score, num_sims, pos, dir]]
        path_queue = [(my_pos, max_step)] # next move options offense
        path_queue_defense = [] # next move options defense
        visited = [adv_pos] # all nodes visited in correct order
        legal = [] # best moves
        legal_defense = [] # defense moves
        cur_step = max_step
        while cur_step - 1:
            if path_queue:
                cur_pos, cur_step = path_queue.pop()
            elif path_queue_defense:
                cur_pos, cur_step = path_queue_defense.pop()
            else:
                break
            x, y = cur_pos
            visited.append(cur_pos)
            for dir, move in enumerate(self.moves):
                if chess_board[x, y, dir]:
                    continue
                #check if gameover, winner = me, winner = adv, tie
                self.set_barrier(x, y, dir, chess_board)
                over, w = self.is_gameover((x, y), adv_pos, chess_board)
                self.remove_barrier(x, y, dir, chess_board)
                if over and w:
                    if w == 1:
                        return [[1, 1, (x, y), dir]], []
                    else:
                        legal_defense.append([-1, 1, (x, y), dir])
                else:
                    # Heuristic: if the direction is right for offense and closer to adv and wall number low (defense), add to front of queue 
                    # else add to back of queue
                    if self.calculate_direction(cur_pos, adv_pos, dir) and sum([chess_board[x, y, i] for i in range(4)]) < 2: 
                        legal.append([0, 1, cur_pos, dir])
                    else:
                        legal_defense.append([0, 1, cur_pos, dir])
                # save next step
                new_x, new_y = x + move[0], y + move[1]
                new_pos = (new_x, new_y)
                if new_pos in visited: # dont walk trhough adv_pos, dont wlak in visited alrdy
                    continue 
                visited.append(new_pos)
                # Heuristic: if the direction is right for offense and closer to adv and wall number low (defense), add to front of queue 
                # else add to back of queue
                if self.calculate_direction(new_pos, adv_pos, dir) and sum([chess_board[new_x, new_y, i] for i in range(4)]) < 2: 
                    path_queue.append((new_pos, cur_step - 1))
                else:
                    path_queue_defense.append((new_pos, cur_step - 1)) 
            if ((len(legal) + len(legal_defense)) > max_step * 4):
                break    
        return legal, legal_defense # NO PRIORITY!
    
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
        move = self.moves[dir]
        step_board[r + move[0], c + move[1], opposites[dir]] = True
    
    def remove_barrier(self, r, c, dir, step_board):
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        # Set the barrier to False
        step_board[r, c, dir] = False
        # Set the opposite barrier to False
        move = self.moves[dir]
        step_board[r + move[0], c + move[1], opposites[dir]] = False
    
    def random_walk(self, my_pos, adv_pos, max_step, chess_board):
        steps = np.random.randint(0, max_step + 1)
        # Pick steps random but allowable moves
        for _ in range(steps):
            r, c = my_pos
            # Build a list of the moves we can make
            allowed_dirs = [ d                                
                for d in range(0,4)                                      # 4 moves possible
                if not chess_board[r,c,d] and                       # chess_board True means wall
                not adv_pos == (r+self.moves[d][0],c+self.moves[d][1])]  # cannot move through Adversary
            if len(allowed_dirs)==0:
                # If no possible move, we must be enclosed by our Adversary
                my_pos = (-1,-1)
                break
            random_dir = allowed_dirs[np.random.randint(0, len(allowed_dirs))]
            # This is how to update a row,col by the entries in moves 
            # to be consistent with game logic
            m_r, m_c = self.moves[random_dir]
            my_pos = (r + m_r, c + m_c)
        if (my_pos==(-1,-1)) :
            #gameover
            #print("PLEASE")
            return r, c, 0
        # Final portion, pick where to put our new barrier, at random
        r, c = my_pos
        # Possibilities, any direction such that chess_board is False
        allowed_barriers=[i for i in range(0,4) if not chess_board[r,c,i]]
        # Sanity check, no way to be fully enclosed in a square, else game already ended
        if (len(allowed_barriers)<1):
            #print("SRLY PLZ")
            return r, c, -1
        assert len(allowed_barriers)>=1 
        dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]
        return r, c, dir
    
    def simulation(self, my_pos, dir, adv_pos, max_step, step_board, start_time):
        # take the current step
        self.set_barrier(my_pos[0], my_pos[1], dir, step_board)
        player_switch = -1 # -1 is adv, 1 is me. every time switch player do player_switch *= -1
        score = 0
        # player = adv
        # random walk
        # check if gameover
        # if gameover, return score
        # else, switch player and return to random walk step (loop)
        # check time somewhere in loop. if time out, ???
        gameover = False
        while (gameover == False) :
            # is_gamover is true if one of the player is enclosed. return bool
            if (player_switch == -1) :
                # adversary's turn
                r, c, dir = self.random_walk(adv_pos, my_pos, max_step, step_board)
                if ((r,c) == (-1,-1) or dir == -1) :
                    gameover, score = self.is_gameover(my_pos, adv_pos, step_board)
                    #print("WHY AM I NOT HERE")
                    break
                #print("simulation adv_pos: ", r, c, dir)
                adv_pos = (r, c)
                self.set_barrier(adv_pos[0], adv_pos[1], dir, step_board) # new wall
            else :
                # my turn
                r, c, dir = self.random_walk(my_pos, adv_pos, max_step, step_board)
                if ((r,c) == (-1,-1) or dir == -1) :
                    #print("WHY AM I NOT HER 2.0")
                    gameover, score = self.is_gameover(my_pos, adv_pos, step_board)
                    break
                my_pos = (r, c)
                #print("simulation my_pos: ", r, c, dir)
                self.set_barrier(my_pos[0], my_pos[1], dir, step_board)
            gameover, score = self.is_gameover(my_pos, adv_pos, step_board)
            player_switch *= -1
            if (self.timeout(start_time)):
                break
        # run random simulation on x, y, d node. if win, s+=1, lose, s-=1. tie s+=0
        return score
    
    def is_gameover(self, my_pos, adv_pos, step_board):
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
        find_my_pos = find(my_pos)
        if adv_pos in find_my_pos:
            return False, 0
        find_adv_pos = find(adv_pos)
        if len(find_adv_pos) == len(find_my_pos):
            return True, 0
        elif len(find_adv_pos) > len(find_my_pos):
            return True, -1
        else:
            return True, 1
    
    def _is_gameover(self, my_pos, adv_pos, step_board): 
        x_max, y_max, _ = step_board.shape
        winner = 0
        # check if gameover, if yes, gameover = True
        # 1 is me, -1 is adv, 0 is tie
        # Union-Find
        father = dict() # init father dict
        for r in range(x_max):
            for c in range(y_max):
                father[(r, c)] = (r, c)

        def find(pos): # Find the root of the node
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2): # set new father
            father[pos1] = pos2

        for r in range(x_max):
            for c in range(y_max):
                for dir, move in enumerate(self.moves[1:3]):  # Only check down and right
                    if step_board[r, c, dir+1]: # If there is a wall, why plus 1???????????????????????????????????????
                        continue
                    pos_a = find((r, c))   # Find the root of r,c node (current node)
                    pos_b = find((r + move[0], c + move[1])) # Find the root of the node after moving
                    if pos_a != pos_b: # If the root of the two nodes are different, union them (set same father)
                        union(pos_a, pos_b)
        p0_r = find(tuple(my_pos)) # Find the root of the player
        p1_r = find(tuple(adv_pos)) # Find the root of the adversary
        p0_score = list(father.values()).count(p0_r) # Count the number of nodes with the same root
        p1_score = list(father.values()).count(p1_r)
        #print("p0_score, p1_score: ", p0_score, p1_score)
        if p0_r == p1_r:
            return False, winner
        if p0_score > p1_score:
            winner = 1
        elif p0_score < p1_score:
            winner = -1
        else:
            winner = 0  # Tie
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

        #children, defense_moves = self.offensive_moves(chess_board, my_pos, adv_pos, max_step)
        children = self.all_moves(chess_board, my_pos, adv_pos, max_step)
        # set my_pos as the root node and do expansions from root
        #next_moves = {}
        # randomly select a child node and do simulations
            # do a number of simulations, return a probability of wins as the node's score
        # continue random exapanions and simulations until time runs out
        #while (self.timeout(start_time) == False):
        if (len(children) == 1): # only one move (usually immediate win)
            my_pos, dir = children[0][3], children[0][4]
            #print("???????????: ", children[0])
        elif (len(children) != 0): 
            for i in range(len(children)):
                while (children[i][2] < self.max_sims and not self.timeout(start_time)): # chess_board.shape[0]
                    step_board = deepcopy(chess_board)
                    score = self.simulation(children[i][3], children[i][4], adv_pos, max_step, step_board, start_time) 
                    children[i][1] += score
                    children[i][2] += 1
                #print("HI: ", children[i], "\n")
                if (self.timeout(start_time)):
                    break
            my_pos, dir = self.best_move(children)
        """
        if (len(children) == 1): # only one move (usually immediate win)
            my_pos, dir = children[0][2], children[0][3]
        elif (len(children) != 0): 
            for i in range(len(children) - 1, 0, -1):
                while (children[i][1] < self.max_sims and not self.timeout(start_time)): # chess_board.shape[0]
                    step_board = deepcopy(chess_board)
                    score = self.simulation(children[i][2], children[i][3], adv_pos, max_step, step_board, start_time) 
                    children[i][0] += score
                    children[i][1] += 1
                print("HI: ", children[i], "\n")
                if (self.timeout(start_time)):
                    break
            my_pos, dir = self.best_move(children)
        
        else : # no good moves, simulate using defense moves
            for i in range(len(defense_moves) - 1, 0, -1):
                while (defense_moves[i][1] < self.max_sims and not self.timeout(start_time)): # chess_board.shape[0]
                    step_board = deepcopy(chess_board)
                    score = self.simulation(defense_moves[i][2], defense_moves[i][3], adv_pos, max_step, step_board, start_time) 
                    defense_moves[i][0] += score
                    defense_moves[i][1] += 1
                print("HI2: ", children[i], "\n")
                if (self.timeout(start_time)):
                    break
            my_pos, dir = self.best_move(defense_moves)
        for i in range(50) :
            x, y, d = self.random_walk(my_pos, adv_pos, max_step, chess_board)
            # check if game is over ???????????????????????????????????????????????????????????
            #print("expansion: x, y, d: ", x, y, d)
            step_board = deepcopy(chess_board)
            self.set_barrier(x, y, d, step_board) # new wall
            s = 0
            num_sim = 0
            for i in range(self.max_sims) :
                num_sim += 1
                s += self.simulation((x, y), d, adv_pos, max_step, step_board)
                # run random simulation on x, y, d node. if win, s+=1, lose, s-=1. tie s+=0
                # score for this node is s/i
                if (self.timeout(start_time)) :
                    # update score as it is and break
                    #print("RIPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")
                    break
            nextmove_key = (x, y, d)
            if (nextmove_key in next_moves):
                next_moves[nextmove_key][0] += s
                next_moves[nextmove_key][1] += num_sim
            else :
                next_moves[nextmove_key] = [s, num_sim]
            if (self.timeout(start_time)) :
                    # update score as it is and break
                    #print("RIPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")
                    break
        # select child with best score and return the move
        best_move = list(next_moves.keys())[0]
        best_score = next_moves[best_move][0]/next_moves[best_move][1]
        for i in next_moves.keys() :
            if (next_moves[i][0]/next_moves[i][1] >= best_score) :
                best_move = i
                best_score = next_moves[i][0]/next_moves[i][1]
        my_pos = (best_move[0], best_move[1])
        dir = best_move[2]
        """

        time_taken = time.time() - start_time     

        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, dir #self.dir_map["u"]
