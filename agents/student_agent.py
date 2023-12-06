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
        self.max_time = 1.9  # max time for each step
        self.max_sims = 4  # max simulations step
        self.max_sels = 3  # max selections for each step
        self.max_node = 10  # max simulation nodes at step 1
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left

    # check if time out
    def timeout(self, start_time):
        return time.time() - start_time > self.max_time

    # calculate the manhattan distance between two positions
    def calculate_distance(self, my_pos, adv_pos):
        r0, c0 = my_pos
        r1, c1 = adv_pos
        return abs(r0 - r1) + abs(c0 - c1)

    # Offensive heuristic: calculate the direction of the adversary
    def calculate_direction(self, my_pos, adv_pos, dir):
        r0, c0 = my_pos
        r1, c1 = adv_pos
        if (r0 < r1 and dir == 2) or (r0 > r1 and dir == 0) or (c0 < c1 and dir == 1) or (c0 > c1 and dir == 3):
            return 1  # if the direction is towards the adversary, p = 1
        elif (r0 == r1 and dir == 2) or (r0 == r1 and dir == 0) or (c0 == c1 and dir == 1) or (c0 == c1 and dir == 3):
            return 0.5  # if the direction is perpendicular to the adversary, p = 0.5
        else:
            return 0  # if the direction is away from the adversary, p = 0

    # build or remove barrier
    def set_barrier(self, r, c, dir, step_board, is_set):
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        # Set the barrier to True/False
        step_board[r, c, dir] = is_set
        # Set the opposite barrier to True/False
        move = self.moves[dir]
        step_board[r + move[0], c + move[1], opposites[dir]] = is_set

    # check if gameover
    def is_gameover(self, my_pos, adv_pos, step_board):
        # return list of spaces reachable from pos using BFS
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

        find_my_pos = find(my_pos)  # list of spaces reachable from my_pos
        if adv_pos in find_my_pos:  # if adv_pos is in find_my_pos the game is not over
            return False, 0
        # game is over, check if I win or lose
        find_adv_pos = find(adv_pos)  # list of spaces reachable from adv_pos
        if len(find_adv_pos) == len(find_my_pos):  # if the two lists are the same, it's a tie
            return True, 0
        elif len(find_adv_pos) > len(find_my_pos):  # if find_adv_pos is larger, I lose
            return True, -1
        else:  # if find_my_pos is larger, I win
            return True, 1

    # get legal moves using A* search and BFS, + heuristic function
    def all_moves(self, chess_board, my_pos, adv_pos, max_step, start_time):
        # do A* search
        [nX, nY, nG, nH, nParent] = [i for i in range(5)]
        open_list = [[my_pos[0], my_pos[1], 0, 0, None]]  # [0: x, 1: y, 2: g, 3: h, 4: parent]
        visited_list = [] # visited nodes (x, y)
        visited_node = [] # visited nodes [p, s, n, (x, y), dir]
        visited = [] # visited nodes [x, y, g, h, parent]
        m_step = 0 # number of steps to the farthest node in the path
        BFS = False # flag to decide if BFS is used
        path_pos = my_pos # in case no path, use current position
        # A* search
        while open_list and not self.timeout(start_time):
            open_list.sort(key=lambda x: x[nG] + x[nH], reverse=True)  # sort on f value
            cur = open_list.pop()  # node with least f value sorted on the rightmost
            x, y = cur[nX], cur[nY] # current position
            if (x, y) == adv_pos: # if reach adv_pos, check if BFS is used
                if BFS: # if BFS is used, continue to next position and finish adding to visited
                    continue
                # backtrack to get the path
                c = cur
                while c[nParent] is not None:  # from end point, until the start point
                    c = visited[c[nParent]]  # go to parent
                    if c[nG] <= max_step: # furthest point on path
                        path_pos = (c[nX], c[nY]) # update path_pos
                        break
                m_step = cur[nG] # number of steps to the endpoint node in the path
                if cur[nG] <= max_step: # if the path is within max_step, use BFS
                    BFS = True
                    continue
                else: # if the path is not within max_step, stop searching 
                    break
            visited_list.append((x, y)) # add to visited_list
            visited.append(cur) # add to visited
            if BFS and cur[nG] > max_step: # if BFS is used, only take nodes within max_step
                continue 
            dp = 1 - cur[nH] / 20 # Heuristic function: offense distance
            # get all children of cur
            for dir, move in enumerate(self.moves):
                if chess_board[x, y, dir]: # if there is a wall, skip the move
                    continue
                if cur[nG] <= max_step: # if the node is within max_step, add to visited_node
                    if BFS and cur[nH] > 2: 
                        continue # only take close enough node when using BFS
                    # check if gameover, me winner = 1, adv winner = -1, tie = 0
                    self.set_barrier(x, y, dir, chess_board, True) # set wall
                    over, p = self.is_gameover((x, y), adv_pos, chess_board) # check if gameover
                    self.set_barrier(x, y, dir, chess_board, False) # remove wall
                    if over:  # gameover, get winner and set p as the returned winner
                        if p == 1: # p = 1 if I win, p = -1 if I lose, p = 0 if tie
                            return [[1, 1, 1, (x, y), dir]]  # immediate win, just play that move
                    else:
                        # Heuristic function: offense distance * defense walls * offense direction
                        p = dp * self.calculate_direction((x, y), adv_pos, dir)
                        # Defense heuristic (counts the number of walls around me)
                        if sum([chess_board[x, y, i] for i in range(4)]) >= 2:  # if there are 2 walls, p = 0.1
                            p *= 0.1 # if there are 3 walls, p *= 0.1
                    visited_node.append([p, 0, 1, (x, y), dir])  # append to legal
                # get the next new node
                new_x, new_y = x + move[0], y + move[1]
                # check if child is in visited_list
                if (new_x, new_y) in visited_list:
                    continue
                # assign x, y, step, g, h, and parent
                new_node = [new_x, new_y, cur[nG] + 1, self.calculate_distance((new_x, new_y), adv_pos), visited.index(cur)]
                # check if child is in open_list with higher steps
                if sum([1 for oN in open_list if ((new_x, new_y) == (oN[nX], oN[nY]) and new_node[nG] > oN[nG])]):
                    continue
                open_list.append(new_node) # add to open_list

        # if no path, use BFS
        if m_step == 0:
            return self.BFS_search(chess_board, my_pos, adv_pos, max_step, start_time)[:self.max_node] # if no path, use BFS
        
        if m_step > max_step * 1.5: # too far away, throw away visited_node
            visited_node = []
        else :
            for n in visited_node:
                if path_pos == n[3]:
                    n[0] += 0.5

        if not visited_node: # only one position - too far
            x, y = path_pos # use the current position
            for d in range(4):
                if not chess_board[x, y, d]:
                    visited_node.append([self.calculate_direction((x, y), adv_pos, d), 0, 1, (x, y), d])

        return sorted(visited_node, key=lambda x: x[0], reverse=True)[:self.max_node] # return top legal moves, based on priority

    # get all legal moves in order of priority based on heuristic function using BFS
    def BFS_search(self, chess_board, my_pos, adv_pos, max_step, start_time):
        # return a list of legal moves using BFS
        # for every possible num of steps until max_step, travel 4 directions + put wall. append to legal, sort and return the top #
        legal = [] # list of [p, s, n, (x, y), dir]
        state_queue = [(my_pos, 0)] # queue of next states
        visited = [my_pos, adv_pos] # list of visited states
        # BFS
        while state_queue and not self.timeout(start_time): # if timeout or state_queue is empty, break
            cur_pos, cur_step = state_queue.pop() # get the current position and current step
            x, y = cur_pos # get the current x, y
            dis_p = 1 - self.calculate_distance(cur_pos, adv_pos) / 20 #get the distance between my current position and adv
            for dir, move in enumerate(self.moves): # 4 directions
                if chess_board[x, y, dir]: # if there is a wall, skip the move
                    continue
                #check if gameover, me winner = 1, adv winner = -1, tie = 0
                self.set_barrier(x, y, dir, chess_board, True)
                over, w = self.is_gameover((x, y), adv_pos, chess_board)
                self.set_barrier(x, y, dir, chess_board, False)
                if over and w: # gameover, get winner and set p as the returned winner
                    p = w # p = 1 if I win, p = -1 if I lose, p = 0 if tie
                    if p == 1:
                        return [[1, 1, 1, (x, y), dir]] #immediate win, just play that move
                else:
                    # Defense heuristic (counts the number of walls around me)
                    if sum([chess_board[x, y, i] for i in range(4)]) >= 2: # if there are 2 walls, p = 0.1 
                        walls = 0.1
                    else: 
                        walls = 1 # if there is 1 wall or no wall, p = 1
                    # Heuristic function: offense distance * defense walls * offense direction
                    p = dis_p * walls * self.calculate_direction((x,y), adv_pos, dir)
                legal.append([p, 0, 1, (x, y), dir]) # append to legal
                # get the next step
                new_x, new_y = x + move[0], y + move[1] 
                new_pos = (new_x, new_y)
                if new_pos not in visited and cur_step + 1 <= max_step: # if not visited and not exceed max_step, append to queue
                    visited.append(new_pos)
                    state_queue.append((new_pos, cur_step + 1))
        return sorted(legal, key=lambda x: x[0], reverse=True) # sort legal moves based on p, higher p first
    
    # Heuristic function: adjust p based on the number of moves left
    def adjust(self, children, my_pos, adv_pos, max_step, chess_board):
        def find_moves(my_pos, adv_pos, chess_board, max_step):
            # return a list of legal moves using BFS
            # for every possible num of steps until max_step, travel 4 directions + put wall. append to legal, sort and return the top #
            n = 0  # list of [(x, y), dir]
            state_queue = [(my_pos, 0)]
            visited = [my_pos, adv_pos]
            # BFS
            while state_queue:
                cur_pos, cur_step = state_queue.pop()
                x, y = cur_pos
                for dir, move in enumerate(self.moves):  # 4 directions
                    if chess_board[x, y, dir]:  # if there is a wall, skip the move
                        continue
                    n += 1  # append to legal
                    # get the next step
                    new_x, new_y = x + move[0], y + move[1]
                    new_pos = (new_x, new_y)
                    if new_pos not in visited and cur_step + 1 <= max_step:  # if not visited and not exceed max_step, append to queue
                        visited.append(new_pos)
                        state_queue.append((new_pos, cur_step + 1))
            return n

        # [(p, s, n, (x, y), dir), ...] -> [0: p, 1: s, 2: n, 3: (x, y), 4: dir]
        for i in range(len(children)):
            # get number of moves left before placing wall
            new_pos, new_dir = children[i][3], children[i][4]
            adv_moves_bef = find_moves(adv_pos, my_pos, chess_board, max_step)
            my_moves_bef = find_moves(my_pos, adv_pos, chess_board, max_step)
            # get number of moves left after placing wall
            self.set_barrier(new_pos[0], new_pos[1], new_dir, chess_board, True)
            adv_moves_aft = find_moves(adv_pos, my_pos, chess_board, max_step)
            my_moves_aft = find_moves(my_pos, adv_pos, chess_board, max_step)
            self.set_barrier(new_pos[0], new_pos[1], new_dir, chess_board, False)
            # update p based on heuristic,
            # offensive: increase p if adv_moves_aft < adv_moves_bef, increase p if my_moves_aft > my_moves_bef
            # defensive: decrease p if adv_moves_aft > adv_moves_bef, decrease p if my_moves_aft < my_moves_bef
            if (adv_moves_aft < adv_moves_bef):
                children[i][0] += 0.5
            if (my_moves_aft > my_moves_bef):
                children[i][0] += 0.5
            if (adv_moves_aft > adv_moves_bef):
                children[i][0] -= 0.5
            if (my_moves_aft < my_moves_bef):
                children[i][0] -= 0.5
        return sorted(children, key=lambda x: x[0], reverse=True) # return top legal moves, based on priority

    # select from the best moves: simulation step + backpropagation step using recursion
    def simulation(self, children, adv_pos, max_step, step_board, start_time, myTurn, depth):
        score, n_sim = 0, 1 # score: total score, n_sim: number of simulations
        for i in range(len(children)): # for each child
            if self.timeout(start_time): 
                break
            n_sim += 1 # increase number of simulations
            my_pos, dir = children[i][3], children[i][4]
            # take the current step
            self.set_barrier(my_pos[0], my_pos[1], dir, step_board, True)
            gameover, gamescore = self.is_gameover(my_pos, adv_pos, step_board)  # check if gameover and get score
            if gameover: # gameover, get score
                if not myTurn: 
                    gamescore = - gamescore # if not myTurn, reverse the score (adv score)
                children[i][1] += gamescore # update score
            elif depth: # not gameover and within depth, continue simulation
                children[i][1] += self.simulation(
                    self.all_moves(step_board, adv_pos, my_pos, max_step, start_time)[:self.max_sels],
                    my_pos, max_step, step_board, start_time, 1 - myTurn, depth - 1) # continue simulations and update score recusively
            self.set_barrier(my_pos[0], my_pos[1], dir, step_board, False) # remove wall
            score += children[i][1] # update total score
        return score / n_sim # return average score

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
        # get all legal moves in order of priority based on heuristic function
        # [(p, s, n, (x, y), dir), ...] -> [0: p, 1: s, 2: n, 3: (x, y), 4: dir]
        children = self.all_moves(chess_board, my_pos, adv_pos, max_step, start_time)  # get top legal moves
        # if only one move just play it - return
        if (len(children) > 1):  # more than one move, run simulations to find the best move
            self.adjust(children, my_pos, adv_pos, max_step, chess_board) # apply heuristic function
            self.simulation(children, adv_pos, max_step, chess_board, start_time, 1, self.max_sims) # run simulations to get scores
            children.sort(key=lambda x: (x[1], x[0]), reverse=True) # sort based on score and then priority
        time_taken = time.time() - start_time
        print("My AI's turn took ", time_taken, "seconds.")
        return children[0][3], children[0][4]