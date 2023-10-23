# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # get distances to each food item
        foodLocations = newFood.asList()
        if len(foodLocations) > 0:
            distToClosestFood = min([util.manhattanDistance(newPos, food) for food in foodLocations])
        else:
            distToClosestFood = 1

        # penalize the evaluation function if the successor state is close to ghost(s)
        # (only if the ghosts are not scared)
        distToGhosts = 0
        for i, state in enumerate(newGhostStates):
            if newScaredTimes[i] == 0:
                distToGhosts += util.manhattanDistance(state.getPosition(), newPos)

        if distToGhosts == 0:
            distToGhosts += 1

        # increment base score by the reciprocal of the distance to the closest food
        # this prioritizes successors that get closer to food
        # also decrement base score by 10 / distance to ghosts
        # this prioritizes moves that get away from ghosts!
        return successorGameState.getScore() + (1 / distToClosestFood) - (10 / distToGhosts)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState, 0)[1]

    def minimax(self, gameState, agentIndex, depth):
        """
        A recursive minimax algorithm. If the agent is Pacman, returns the maximum
        minimax score and associated action. If the agent is a ghost, returns the 
        score of the minimum scored action.
        """
        # check for terminal state or depth
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        
        if agentIndex == 0:  # Pacman: max
            return self.maxValue(gameState, depth)[0]
        else:  # Ghost: min
            return self.minValue(gameState, agentIndex, depth)
        
    def maxValue(self, gameState, depth):
        """
        Returns the maximum score and associated action among Pacman's legal actions.
        """
        v = float('-inf')  # initialize v as negative infinity

        for action in gameState.getLegalActions(0):
            # get the successor state
            successor = gameState.generateSuccessor(0, action)
            # v becomes the max of v and the minimax function called on the successor
            # save the best action!
            v_new = self.minimax(successor, 1, depth)
            if v_new > v:
                v = v_new
                best_action = action
        return v, best_action
    
    def minValue(self, gameState, agentIndex, depth):
        """
        Returns the score of the minimum scored action among the agent (ghost)'s legal actions.
        """
        v = float('inf')  # initialize v as infinity

        for action in gameState.getLegalActions(agentIndex):
            # get the successor state
            successor = gameState.generateSuccessor(agentIndex, action)
            # v becomes the min of v and the minimax function called on the successor
            if agentIndex + 1 == gameState.getNumAgents():
                v = min(v, self.minimax(successor, 0, depth + 1))
            else:
                v = min(v, self.minimax(successor, agentIndex + 1, depth))
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        "*** YOUR CODE HERE ***"
        alpha = float('-inf')  # initialize alpha
        beta = float('inf')  # initialize beta
        return self.maxValue(gameState, 0, alpha, beta)[1]

    def minimax(self, gameState, agentIndex, depth, alpha, beta):
        """
        Recursive minimax implementation with alpha-beta pruning.
        """
        # check for terminal state or depth
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        
        if agentIndex == 0:  # Pacman: max
            return self.maxValue(gameState, depth, alpha, beta)[0]
        else:  # Ghost: min
            return self.minValue(gameState, agentIndex, depth, alpha, beta)
        
    def maxValue(self, gameState, depth, alpha, beta):
        """
        Returns the maximum score and associated action among Pacman's legal actions.
        If the value is greater than beta, returns that value.
        """
        v = float('-inf')  # initialize v as negative infinity

        for action in gameState.getLegalActions(0):
            # get the successor state
            successor = gameState.generateSuccessor(0, action)
            # v becomes the max of v and the minimax function called on the successor
            # save the best action!
            v_new = self.minimax(successor, 1, depth, alpha, beta)
            if v_new > v:
                v = v_new
                best_action = action
            if v > beta:  # test against beta
                return v, best_action
            alpha = max(alpha, v)  # prune alpha
        return v, best_action
    
    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        """
        Returns the minimum score among the agent's legal actions.
        If the value is less than alpha, returns that value.
        """
        v = float('inf')  # initialize v as infinity

        for action in gameState.getLegalActions(agentIndex):
            # get the successor state
            successor = gameState.generateSuccessor(agentIndex, action)
            # v becomes the min of v and the minimax function called on the successor
            if agentIndex + 1 == gameState.getNumAgents():
                v = min(v, self.minimax(successor, 0, depth + 1, alpha, beta))
            else:
                v = min(v, self.minimax(successor, agentIndex + 1, depth, alpha, beta))
            if v < alpha:  # test against alpha
                return v
            beta = min(v, beta)  # prune beta
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState, 0)[1]  # get pacman's code

    def expectimax(self, gameState, agentIndex, depth):
        """
        Recursive expectimax implementation. Returns the max scored action for
        Pacman or the expected value of the score of the ghost's action.
        """
        # check for terminal state or depth
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        
        if agentIndex == 0:  # Pacman: max
            return self.maxValue(gameState, depth)[0]
        else:  # Ghost: min
            return self.expectedValue(gameState, agentIndex, depth)
        
    def maxValue(self, gameState, depth):
        """
        Returns the maximum score and associated action among Pacman's legal actions.
        """
        v = float('-inf')  # initialize v as negative infinity

        for action in gameState.getLegalActions(0):
            # get the successor state
            successor = gameState.generateSuccessor(0, action)
            # v becomes the max of v and the expectimax function called on the successor
            # save the best action!
            v_new = self.expectimax(successor, 1, depth)
            if v_new > v:
                v = v_new
                best_action = action
        return v, best_action
    
    def expectedValue(self, gameState, agentIndex, depth):
        """
        Returns the expected value of the score of the ghost's action.
        """
        expected = 0  # initialize storage for expected score

        if not gameState.getLegalActions():  # leaf, so return eval function
            return self.evaluationFunction(gameState)

        # assume that the ghost chooses an action at random from uniform
        prob = 1 / len(gameState.getLegalActions(agentIndex))

        for action in gameState.getLegalActions(agentIndex):
            # get the successor state
            successor = gameState.generateSuccessor(agentIndex, action)

            # get the score of that successor using expectimax
            if agentIndex + 1 == gameState.getNumAgents():
                score = self.expectimax(successor, 0, depth + 1)
            else:
                score = self.expectimax(successor, agentIndex + 1, depth)

            # increment the expected value:
            expected += prob * score
        return expected

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    Increments the current score by the reciprocal of the distance to the
    closest food item and 10 * the reciprocal of the total distance to the
    ghost. If the ghosts are scared, the total distance to the ghosts is 
    negated.
    """

    "*** YOUR CODE HERE ***"
    # considerations:
    # base motivation: get to the nearest pellet
    # if ghosts are scared, prioritize actions closer to ghost
    # if ghosts are not scared, prioritize actions away from ghosts

    ###
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    # get distances to each food item
    foodLocations = newFood.asList()
    if len(foodLocations) > 0:
        distToClosestFood = min([util.manhattanDistance(newPos, food) for food in foodLocations])
    else:
        distToClosestFood = 1  # to avoid dividing by zero

    # get total distance to each ghost
    distToGhosts = 0
    for state in newGhostStates:
        distToGhosts += util.manhattanDistance(state.getPosition(), newPos)

    if distToGhosts == 0:
        distToGhosts += 1  # to avoid dividing by zero

    # reverse incentive if ghosts are scared:
    if newScaredTimes[0] > 0:
        distToGhosts *= -1

    # increment base score by the reciprocal of the distance to the closest food
    # this prioritizes successors that get closer to food
    # also decrement base score by 10 / distance to ghosts
    # this prioritizes moves that get away from ghosts! or moves that approach scared ghosts
    return currentGameState.getScore() + (1 / distToClosestFood) - (10 / distToGhosts)


# Abbreviation
better = betterEvaluationFunction
