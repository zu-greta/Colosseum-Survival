# mcts 2 but class + engame check + heuristics
# win rate 100! but 14secs per move

# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time

from collections import deque




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
        self.max_time = 13.9 # max time per move
        self.m = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def timeout(self, start_time):
        return time.time() - start_time > self.max_time
        
    class MCTS():
        def __init__(self, score, num_sims, parent, pos, dir, priority):
            self.score = score
            self.num_sims = num_sims
            self.parent = parent
            self.pos = pos
            self.dir = dir
            self.priority = priority

            self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
            self.max_time = 13.9 # max time per move
        
        def update(self, score):
            self.score += score
            self.num_sims += 1

        def get_score(self):
            if self.num_sims == 0:
                return 0
            return self.score / self.num_sims
        
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
    
        #TODO: implement legal_moves - change it up
        def legal_moves(self, chess_board, my_pos, adv_pos, max_step, root):
            # return a list of legal moves using BFS
            # for every possible num of steps until max_step, travel 4 directions + put wall. append to legal and return
            legal = []
            state_queue = [(my_pos, 0)]
            visited = [(my_pos)]
            #distance = self.calculate_distance(my_pos, adv_pos) # get distance between me and adv
            # BFS
            while state_queue:
                cur_pos, cur_step = state_queue.pop(0) #get next element
                x, y = cur_pos #get the position in x, y coordinate
                dis = self.calculate_distance(cur_pos, adv_pos) #get the distance between my current position and adv
                for dir, move in enumerate(self.moves): #for each direction
                    if chess_board[x, y, dir]: #if there is a wall, dont walk thruough it
                        continue
                    #check if gameover, winner = me, winner = adv, tie
                    self.set_barrier(x, y, dir, chess_board) #set the barrier
                    #print("barrier set: ", chess_board[x, y, dir])
                    over, w = self.is_gameover((x, y), adv_pos, chess_board)
                    self.remove_barrier(x, y, dir, chess_board) #remove the barrier
                    if over and w:
                        p = w 
                        if p == 1: #if i win, return the winning move (dont care about the rest)
                            return [StudentAgent.MCTS(1, 1, root, (x, y), dir, p)] 
                        #elif len(legal) > 0: # only save one losing move just in case
                        #    continue
                    else:
                        if sum([chess_board[x, y, i] for i in range(4)]) >= 2: 
                            walls = 0 
                        else: 
                            walls = 1 #if there are 2 walls, dont put another one 
                        # heuristics: distance (be aggressive), walls (be defensive), direction of wall (be both)
                        p = (1 - dis/20) * walls * self.calculate_direction((x,y), adv_pos, dir) 
                    new_node = StudentAgent.MCTS(0, 0, root, (x, y), dir, p)
                    #if len(legal) == 0: #if no legal moves yet, append
                    #    legal.append(new_node)
                    #elif legal[0].priority < p: #if new node better, put at the front
                    #    legal.insert(0, new_node)
                    #else: #else put at the back
                    legal.append(new_node)
                    new_x, new_y = x + move[0], y + move[1]
                    new_pos = (new_x, new_y)
                    if new_pos == adv_pos or new_pos in visited: #if the new position is the adversary or we have visited it
                        continue
                    if cur_step + 1 <= max_step: #if we are not outside of the step range
                        visited.append(new_pos)
                        state_queue.append((new_pos, cur_step + 1))
            # return list of child nodes (MCTS objects - 0/0 with parent = root)
            return sorted(legal, key=lambda x: x.priority, reverse=True)      

        # possible improvement: instead of complete random walk - choose with heuristic (eg. dont close myself)
        def random_walk(self, my_pos, adv_pos, max_step, chess_board):
            steps = np.random.randint(0, max_step + 1)
            # pos_pos = my_pos
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
        
        #TODO: simulate as player and not random
        #TODO: bring in which sim its at and randomly choose a number from first few to simulate
        def simulate(self, my_pos, adv_pos, max_step, step_board, start_time):
            # take the current step
            my_pos = self.pos
            self.set_barrier(my_pos[0], my_pos[1], self.dir, step_board) # new wall
            player_switch = -1 # -1 is adv, 1 is me. every time switch player do player_switch *= -1
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
                    adv_cur = StudentAgent.MCTS(0, 0, None, adv_pos, None, None)
                    a_children = adv_cur.legal_moves(step_board, adv_pos, my_pos, max_step, adv_cur)
                    if len(a_children) == 0:
                        gameover, score = self.is_gameover(my_pos, adv_pos, step_board)
                        break
                    else:
                        n = np.random.randint(0, min(5, len(a_children)))
                        adv_pos, dir = a_children[n].pos, a_children[n].dir
                        self.set_barrier(adv_pos[0], adv_pos[1], dir, step_board) # new wall
                else :
                    # my turn
                    my_cur = StudentAgent.MCTS(0, 0, None, my_pos, None, None)
                    children = my_cur.legal_moves(step_board, my_pos, adv_pos, max_step, my_cur)
                    if len(children) == 0:
                        gameover, score = self.is_gameover(my_pos, adv_pos, step_board)
                        break
                    else:
                        n = np.random.randint(0, min(5, len(children)))
                        my_pos, dir = children[n].pos, children[n].dir
                        self.set_barrier(my_pos[0], my_pos[1], dir, step_board) # new wall
                gameover, score = self.is_gameover(my_pos, adv_pos, step_board)
                player_switch *= -1
                if (StudentAgent.timeout(self, start_time)):
                    break
            # run random simulation on x, y, d node. if win, s+=1, lose, s-=1. tie s+=0
            return score
        
        def is_gameover(self, my_pos, adv_pos, step_board): 
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
                        if step_board[r, c, dir + 1]: # If there is a wall, why plus 1???????????????????????????????????????
                            continue
                        pos_a = find((r, c))   # Find the root of r,c node (current node)
                        pos_b = find((r + move[0], c + move[1])) # Find the root of the node after moving
                        if pos_a != pos_b: # If the root of the two nodes are different, union them (set same father)
                            union(pos_a, pos_b)
            p0_r = find(tuple(my_pos)) # Find the root of the player
            p1_r = find(tuple(adv_pos)) # Find the root of the adversary
            p0_score = list(father.values()).count(p0_r) # Count the number of nodes with the same root
            p1_score = list(father.values()).count(p1_r)
            if p0_r == p1_r:
                return False, winner
            if p0_score > p1_score:
                winner = 1
            elif p0_score < p1_score:
                winner = -1
            else:
                winner = 0  # Tie
            return True, winner
        
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
        
        
        def best_move(self, legal_moves):
            # return a tuple of ((x, y), dir),
            # where (x, y) is the next position of your agent and dir is the direction of the wall you want to put on.
            best_next_move = sorted(legal_moves, key=lambda x: (x.get_score(), x.priority), reverse=True)[0]
            return best_next_move.pos, best_next_move.dir
        

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

        # set my_pos as the root node and do expansions from root
        root = self.MCTS(0, 0, None, my_pos, None, None)
        children = root.legal_moves(chess_board, my_pos, adv_pos, max_step, root) # get list of legal moves
        print("children outside: ", children[0].pos, children[0].dir, children[0].priority, children[0].get_score(), children[0].num_sims)
        if children[0].priority == 1 or children[0].priority == -1: #can win or immediate loss the game right away
            my_pos, dir = children[0].pos, children[0].dir
        # else run simulations and choose best move
        else:
            for i in range(len(children)):
                if children[i].priority == -1:
                    continue # skip the bad move #don't continue looking at losing moves
                while (children[i].num_sims < 20 and not self.timeout(start_time)): # chess_board.shape[0]
                    step_board = deepcopy(chess_board)
                    score = children[i].simulate(my_pos, adv_pos, max_step, step_board, start_time) 
                    children[i].update(score)
                print("HI?", children[i].pos, children[i].dir, children[i].priority, children[i].get_score(), children[i].num_sims)
                if (self.timeout(start_time)):
                    break
            print("children: ", children[i].pos, children[i].dir, children[i].priority, children[i].get_score(), children[i].num_sims)
            my_pos, dir = root.best_move(children)      
        time_taken = time.time() - start_time     
        print("My AI's turn took ", time_taken, "seconds.")
        return my_pos, dir # return the best move
