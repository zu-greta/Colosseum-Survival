# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time

from collections import deque

@register_agent("student21_agent")
class Student21Agent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """
    # Extra parameters were used to to allow for genetic algorithm training
    def __init__(self, expansionWeight=7.614144709985369, agressiveWeight=1.7574868125047338, centerDistanceWeight=8.200845538225007, openSpaceWeight=5.33281289566942, extendBarrierWeight=8.602283333747607):
        super(Student21Agent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

        self.maxStep = 0
        self.startTime = None
        self.boardSize = 0
        self.cutoffTime = 1.85

        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        self.transpositionTable = {}

        self.expansionWeight = expansionWeight
        self.agressiveWeight = agressiveWeight
        self.centerDistanceWeight = centerDistanceWeight
        self.openSpaceWeight = openSpaceWeight
        self.extendBarrierWeight = extendBarrierWeight

    def step(self, chess_board, myPos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - myPos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next pos of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds
        # 
        # Add iterative deepening
        # Move ordering
        # Transpos
        self.startTime = time.time()
        bestScore = float("-inf")
        bestMove = None
        depth = 1

        if self.maxStep == 0:
            self.maxStep = max_step
        if self.boardSize == 0:
            self.boardSize = chess_board.shape[0]

        while True:
            score, move = self.alphaBeta(myPos, adv_pos, depth, chess_board)
            currentTime = time.time()
            
            if score > bestScore:
                bestMove = move 
                bestScore = score

            if currentTime - self.startTime > self.cutoffTime:
                break  
            depth += 1

        return bestMove
    
    def doMove(self, pos, direction, chessBoard):

        posX, posY = pos
        adjacentX, adjacentY = (posX + self.moves[direction][0], posY + self.moves[direction][1])
        chessBoard[posX, posY, direction] = True
        chessBoard[adjacentX, adjacentY, self.opposites[direction]] = True

    def undoMove(self, pos, direction, chessBoard):

        posX, posY = pos
        adjacentX, adjacentY = (posX + self.moves[direction][0], posY + self.moves[direction][1])
        chessBoard[posX, posY, direction] = False
        chessBoard[adjacentX, adjacentY, self.opposites[direction]] = False

    def alphaBeta(self, myPos, advPos, depth, chessBoard):

        alpha = float("-inf")
        beta = float("inf")
        return self.MaxValue(myPos, advPos, depth, chessBoard, alpha, beta)

    def MaxValue(self, myPos, advPos, depth, chessBoard, alpha, beta, evalMove = None):
        
        isGameOver, myScore, advScore = self.isGameOver(myPos, advPos, chessBoard)
        if self.cutoff(myPos, advPos, depth, chessBoard) or isGameOver:
            return self.eval(myPos, advPos, chessBoard, evalMove, myScore - advScore), None

        # Transposition implementation for more efficient iterative deepening, based on: http://people.csail.mit.edu/plaat/mtdf.html#abmem
        transpositionKey = self.getTranspositionKey(myPos, advPos, chessBoard, True)
        if transpositionKey in self.transpositionTable:

            entry = self.transpositionTable[transpositionKey]
            if entry["depth"] >= depth:
                if entry["flag"] == "exact":
                    return entry["score"], entry["bestMove"]
                elif entry["flag"] == "lowerbound":
                    alpha = max(alpha, entry["score"])
                elif entry["flag"] == "upperbound":
                    beta = min(beta, entry["score"])
                if alpha >= beta:
                    return entry["score"], entry["bestMove"]

        maxScore = float("-inf")
        bestMove = None
        legalMoves = self.getLegalMoves(myPos, advPos, chessBoard)
        
        for move in legalMoves:
            self.doMove(move[0], move[1], chessBoard)
            score, _ = self.MinValue(move[0], advPos, depth - 1, chessBoard, alpha, beta, move)
            self.undoMove(move[0], move[1], chessBoard)

            if score > maxScore:
                maxScore = score
                bestMove = move
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        # Save to transposition table
        flag = "exact" if maxScore <= alpha else "lowerbound"
        self.transpositionTable[transpositionKey] = {"score": maxScore, "depth": depth, "bestMove": bestMove, "flag": flag}
        
        return maxScore, bestMove

    def MinValue(self, myPos, advPos, depth, chessBoard, alpha, beta, evalMove = None):

        isGameOver, myScore, advScore = self.isGameOver(myPos, advPos, chessBoard)
        if self.cutoff(myPos, advPos, depth, chessBoard) or isGameOver:
            return self.eval(myPos, advPos, chessBoard, evalMove, myScore - advScore), None

        # Transposition implementation for more efficient iterative deepening, based on: http://people.csail.mit.edu/plaat/mtdf.html#abmem
        transpositionKey = self.getTranspositionKey(myPos, advPos, chessBoard, False)
        if transpositionKey in self.transpositionTable:

            entry = self.transpositionTable[transpositionKey]
            if entry["depth"] >= depth:
                if entry["flag"] == "exact":
                    return entry["score"], entry["bestMove"]
                elif entry["flag"] == "lowerbound":
                    alpha = max(alpha, entry["score"])
                elif entry["flag"] == "upperbound":
                    beta = min(beta, entry["score"])
                if alpha >= beta:
                    return entry["score"], entry["bestMove"]

        minScore = float("inf")
        bestMove = None

        legalMoves = self.getLegalMoves(advPos, myPos, chessBoard)

        for move in legalMoves:

            self.doMove(move[0], move[1], chessBoard)
            score, _ = self.MaxValue(myPos, move[0], depth - 1, chessBoard, alpha, beta, move)
            self.undoMove(move[0], move[1], chessBoard)
            if score < minScore:
                minScore = score
                bestMove = move
            beta = min(beta, score)
            if alpha >= beta:
                return minScore, bestMove
            
        # Save to transposition table
        flag = "exact" if minScore >= beta else "upperbound"
        self.transpositionTable[transpositionKey] = {"score": minScore, "depth": depth, "bestMove": bestMove, "flag": flag}

        return minScore, bestMove

    # Obtain transpositionkey
    def getTranspositionKey(self, myPos, advPos, chessBoard, isMaximizing):
        return (myPos, advPos, chessBoard.tostring(), isMaximizing)
    
    # Cutoff when conditions are met
    def cutoff(self, myPos, advPos, depth, chessBoard):
        current_time = time.time()
        return current_time - self.startTime > self.cutoffTime or depth == 0

    # Optimized checkEndGame function based on world.py, and reformatted for better readability
    def isGameOver(self, myPos, advPos, chessBoard):
        
        # Added path compression and size tracking
        parent = dict()
        size = dict()
        for r in range(self.boardSize):
            for c in range(self.boardSize):
                parent[(r, c)] = (r, c)
                size[(r, c)] = 1

        def find(pos):
            if parent[pos] != pos:
                parent[pos] = find(parent[pos])
            return parent[pos]

        def union(pos1, pos2):
            root1, root2 = find(pos1), find(pos2)
            if root1 != root2:
                if size[root1] < size[root2]:
                    root1, root2 = root2, root1 
                parent[root2] = root1
                size[root1] += size[root2]

        # Only check down and right
        directions = [(0, 1), (1, 0)]  # Right, Down
        for r in range(self.boardSize):
            for c in range(self.boardSize):
                for move in directions:
                    if chessBoard[r, c, 1 if move == (0, 1) else 2]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        # Find roots for each player
        p0_r = find(tuple(myPos))
        p1_r = find(tuple(advPos))

        # Scores are now directly available from size dictionary
        p0_score = size[p0_r]
        p1_score = size[p1_r]

        # Check if players belong to the same set
        gameOverResult = p0_r != p1_r
        return gameOverResult, p0_score, p1_score
    
    # Check if move is within boundary
    def checkBoundary(self, pos):
        x, y = pos
        return 0 <= x < self.boardSize and 0 <= y < self.boardSize
    
    # Returns all legal moves
    def getLegalMoves(self, myPos, advPos, chessBoard):

        legalMoves = set()
        visited = set()
        queue = deque([(myPos, self.maxStep)])

        while queue:
            currentPos, stepsLeft = queue.popleft()
            if currentPos in visited or not self.checkBoundary(currentPos) or currentPos == advPos:
                continue
            visited.add(currentPos)

            for directionIndex, (deltaX, deltaY) in enumerate(self.moves):
                if not chessBoard[currentPos[0]][currentPos[1]][directionIndex]:
                    nextPosition = (currentPos[0] + deltaX, currentPos[1] + deltaY)
                    if nextPosition not in visited:
                        if stepsLeft > 0:
                            queue.append((nextPosition, stepsLeft - 1))
                    legalMoves.add((currentPos, directionIndex))
        return list(legalMoves)
    
    # Heuristic evaluation function
    def eval(self,myPos, advPos, chessBoard, move, endScore):
        score = 0
        totalWalls = np.sum(chessBoard)
        winLose = self.didWin(endScore)

        if winLose == None:
            score += self.expansionHeuristic(myPos, advPos, chessBoard) * self.expansionWeight
            score += self.centerDistanceHeuristic(chessBoard, myPos) * self.centerDistanceWeight
            score += self.openSpaceHeuristic(chessBoard, myPos) * self.openSpaceWeight
            score += self.aggresiveHeuristic(myPos, advPos, totalWalls) * self.agressiveWeight
            if move != None:
                score += self.extendBarrierHeuristic (chessBoard, move) * self.extendBarrierWeight
        elif winLose:
            return 9999
        else:
            return -9999

        return score

    # Determines if the move is winning or losing
    def didWin(self, endScore):
        if endScore > 0:
            return True
        elif endScore < 0:
            return False
        else:
            return None
    
    # Based on number of moves available
    def expansionHeuristic(self, myPos, advPos, chessBoard):
        myMoves = len(self.getLegalMoves(myPos, advPos, chessBoard))
        advMoves = len(self.getLegalMoves(advPos, myPos, chessBoard))
        return (myMoves - advMoves)
    
    # Makes the agent more agressive in the early game
    def aggresiveHeuristic(self, myPos, advPos, totalWalls):
        distanceToAdv = abs(myPos[0] - advPos[0]) + abs(myPos[1] - advPos[1])
        score = 0
        maxWalls = self.boardSize * (self.boardSize - 1) * 2  # Maximum possible walls

        # Calculate the percentage of walls placed
        wallPercentage = (totalWalls / maxWalls) * 100
        if wallPercentage < 50:
            score -= distanceToAdv

        return score
    
    # Penalizes being further away than the center
    def centerDistanceHeuristic(self,chessBoard, myPos):
        x, y = myPos
        centerPos = len(chessBoard) / 2
        return -(abs(x - centerPos) + abs(y - centerPos))

    # Rewards open space 
    def openSpaceHeuristic(self,chessBoard, myPos):
        x, y = myPos
        totalSquares = 0
        totalWalls = 0
        for x in range(max(0, x - 1), min(len(chessBoard), x + 1)):
            for y in range(max(0, y - 1), min(len(chessBoard), y + 1)):
                totalSquares += 1
                for direction in range(0, 4):
                    if chessBoard[x, y, direction]:
                        totalWalls += 1
        return (totalSquares - totalWalls) / totalSquares
    
    # Rewards moves that can extend a barrier
    def extendBarrierHeuristic(self, chessBoard, move):
        score = 0
        x, y, direction = move[0][0], move[0][1], move[1]

        # Check for extending barriers in the same direction as the move
        if direction == 0:  # up
            if x > 0 and chessBoard[x - 1, y, 0]: 
                score += 1
        elif direction == 1:  # right
            if y < chessBoard.shape[1] - 1 and chessBoard[x, y + 1, 1]:  
                score += 1
        elif direction == 2:  # down
            if x < chessBoard.shape[0] - 1 and chessBoard[x + 1, y, 2]:  
                score += 1
        elif direction == 3:  # left
            if y > 0 and chessBoard[x, y - 1, 3]:  
                score += 1

        return score