# copy2
# 50 expansions + 10 simulations
# 0.92-8 success against random agent


# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time





@register_agent("student2_agent")
class Student2Agent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(Student2Agent, self).__init__()
        self.name = "Student2Agent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.max_time = 1.9
        self.max_sim = 10
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))


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
        """
        print("allowed_barriers: ", allowed_barriers)
        print("pos_pos: ", my_pos)
        print("board: ", chess_board[r, c, 0])
        print("board: ", chess_board[r, c, 1])
        print("board: ", chess_board[r, c, 2])
        print("board: ", chess_board[r, c, 3])
        """
        if (len(allowed_barriers)<1):
            #print("SRLY PLZ")
            return r, c, -1
        assert len(allowed_barriers)>=1 
        dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]
        return r, c, dir

    def timeout(self, start_time):
        return time.time() - start_time > self.max_time
    
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
    
    #check valid step????

    def simulation(self, my_pos, dir, adv_pos, max_step, step_board):
        # take the current step
        self.set_barrier(my_pos[0], my_pos[1], dir, step_board)
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
        next_moves = {}
        # randomly select a child node and do simulations
            # do a number of simulations, return a probability of wins as the node's score
        # continue random exapanions and simulations until time runs out
        #while (self.timeout(start_time) == False):
        for i in range(50) :
            x, y, d = self.random_walk(my_pos, adv_pos, max_step, chess_board)
            # check if game is over ???????????????????????????????????????????????????????????
            #print("expansion: x, y, d: ", x, y, d)
            step_board = deepcopy(chess_board)
            self.set_barrier(x, y, d, step_board) # new wall
            s = 0
            num_sim = 0
            for i in range(self.max_sim) :
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

        time_taken = time.time() - start_time     

        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, dir #self.dir_map["u"]
