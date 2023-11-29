# mcts with UCB - copy3
# 0.45 or 0.76? win rate - 0 T^T

# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time

from collections import deque




@register_agent("student3_agent")
class Student3Agent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(Student3Agent, self).__init__()
        self.name = "Student3Agent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.max_time = 1.9
        self.max_sim = 20 # number of simulations per child

    def timeout(self, start_time):
        return time.time() - start_time > self.max_time
    
    class MCTS():
        def __init__(self, score, num_sims, parent, pos, dir):
            self.score = score
            self.num_sims = num_sims
            self.parent = parent
            self.pos = pos
            self.dir = dir

            self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
            self.c = 1.41421356237 # exploration constant
        
        def update(self, score):
            self.score += score
            self.num_sims += 1

        def get_score(self):
            if self.num_sims == 0:
                return 0
            return self.score / self.num_sims
        
        #TODO: implement legal_moves - change it up
        def legal_moves(self, chess_board, my_pos, adv_pos, max_step, root):
            # return a list of legal moves using BFS
            # for every possible num of steps until max_step, travel 4 directions + put wall. append to legal and return
            legal = []
            legal_bad = []
            state_queue = [(my_pos, 0)]
            visited = [(my_pos)]
            # BFS
            while state_queue:
                cur_pos, cur_step = state_queue.pop(0) #get next element
                if cur_step == max_step:
                    break #if we are not outside of the step range
                x, y = cur_pos #get the position in x, y coordinate
                for dir, move in enumerate(self.moves): #for each direction
                    if chess_board[x, y, dir]: #if there is a wall, dont walk thruough it
                        continue
                    walls = 0 #count the number of walls
                    for wall in range(4): #for each wall
                        if chess_board[x, y, wall]: #if there is a wall
                            walls += 1 #add to the wall count
                    if walls < 3:
                        # create a node to append
                        new_node = Student3Agent.MCTS(0, 0, root, (x, y), dir)
                        legal.append(new_node) #append the legal move
                    else: 
                        new_node = Student3Agent.MCTS(0, 0, root, (x, y), dir)
                        legal_bad.append(new_node) #append the legal move
                    new_x, new_y = x + move[0], y + move[1]
                    new_pos = (new_x, new_y)
                    if new_pos == adv_pos or new_pos in visited:
                        continue
                    visited.append(new_pos)
                    state_queue.append((new_pos, cur_step + 1))
            # return list of child nodes (MCTS objects - 0/0 with parent = root)
            return legal, legal_bad
        
        """
        #TODO: improve select_child - heuristic/UCB
        def select_child(self, children):
            # return a child node using UCB, heuristic or random
            n = np.random.randint(0, len(children)) # randomly choose a child to select
            child = children[n]
            return child
        
        """
        #UCB version
        def select_child(self, children):
            best_next_move = children[-1]
            for move in children:
                if move.UCB() > best_next_move.UCB():
                    best_next_move = move
            return best_next_move
        
        
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
        
        def expand(self, my_pos, adv_pos, max_step, chess_board):
            # return a child node (MCTS object - 0/0 with parent = self)
            # do a random_walk and return it as the node
            x, y, dir = self.random_walk(my_pos, adv_pos, max_step, chess_board)
            pos = (x, y)
            node = Student3Agent.MCTS(0, 0, self, pos, dir)
            return node

        def simulate(self, my_pos, adv_pos, max_step, step_board):
            player_switch = -1 # -1 is sdv, 1 is me. every time switch player do player_switch *= -1
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
            # what's this for ??
            #for r in range(x_max):
            #    for c in range(y_max):
            #        find((r, c)) # Find the root of each node
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
        
        """
        # 0.49 win 
        def best_move(self, legal_moves):
            # return a tuple of ((x, y), dir),
            # where (x, y) is the next position of your agent and dir is the direction of the wall you want to put on.
            best_next_move = legal_moves[-1]
            #print("best_next_move: ", best_next_move.pos, best_next_move.dir)
            #print("best_next_move score: ", best_next_move.get_score())
            for move in legal_moves:
                #print("move: ", move.pos, move.dir)
                #print("move score: ", move.get_score())
                if move.get_score() > best_next_move.get_score():
                    best_next_move = move
            pos, dir = best_next_move.pos, best_next_move.dir
            return pos, dir
        """
        
        # division by 0?????
        def UCB(self):
            # return the UCB value of the node
            if self.num_sims == 0:
                return 0
            else: 
                return self.get_score() + self.c * np.sqrt(np.log(self.parent.num_sims) / self.num_sims)

        def best_move(self, legal_moves):
            best_next_move = legal_moves[-1]
            for move in legal_moves:
                if move.UCB() > best_next_move.UCB():
                    best_next_move = move
            pos, dir = best_next_move.pos, best_next_move.dir
            return pos, dir
        
        

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
        step_board = deepcopy(chess_board)
        root = self.MCTS(0, 0, None, my_pos, None)
        good_children, bad_children = root.legal_moves(step_board, my_pos, adv_pos, max_step, root) #TODO: change-up legal_moves
        #TODO: get rid of some of the bad children ???
        num_childs = 0
        while(num_childs < len(good_children) and not self.timeout(start_time)):
        #for i in range(len(children)):
            # selection of a good child - either hueristic, random or UCB
            if len(good_children) != 0:
                child = root.select_child(good_children) #TODO: improve select_child
                #print("good child")
            else:
                child = root.select_child(bad_children)
                #print("bad child")
            num_sims = 0
            while(num_sims < self.max_sim and not self.timeout(start_time)):
                # random exanpanions from child
                node = child.expand(my_pos, adv_pos, max_step, step_board) 
                score = node.simulate(my_pos, adv_pos, max_step, step_board) 
                node.update(score)
                node.parent.update(score)
                #print("node score: ", node.get_score())
                #print("node.parent score: ", node.parent.get_score())
                num_sims += 1
            num_childs += 1
        #select child with best score and return the move
        children = good_children + bad_children
        best_next_move = root.best_move(children) 
        my_pos, dir = best_next_move
        
        time_taken = time.time() - start_time     

        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, dir #self.dir_map["u"]
