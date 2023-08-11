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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        ghostDistances = [util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        nearestGhostDistance = min(ghostDistances)
        
        foodDistances = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
        if foodDistances:
            nearestFoodDistances = min(foodDistances)
        else:
            nearestFoodDistances = 0
        
        nearestGhostIndex = ghostDistances.index(nearestGhostDistance)
        nearestGhostScaredTime = newScaredTimes[nearestGhostIndex]
        ghostScore = -nearestFoodDistances if nearestGhostScaredTime > 0 else nearestGhostDistance

        ghostPositions = list(ghostState.getPosition() for ghostState in newGhostStates)
        
        foodDistancesToGhosts = []
        for food in newFood.asList():
            distances = [util.manhattanDistance(food,ghost) for ghost in ghostPositions]
            if distances:
                foodDistancesToGhosts.append(min(distances))
            else:    
                foodDistancesToGhosts.append(float('inf'))
        if not foodDistancesToGhosts:
            return float('inf')
        safeFoodDistance = [util.manhattanDistance(newPos,food) for food in newFood.asList() if (min([util.manhattanDistance(food,ghost) for ghost in ghostPositions]) if ghostPositions else float('inf')) == max(foodDistancesToGhosts)]
        minSafeFoodDistance = min(safeFoodDistance) if safeFoodDistance else 0

        legalActions = successorGameState.getLegalActions()
        futureStates = [successorGameState.generatePacmanSuccessor(a) for a in legalActions]
        bestFutureState = float('inf')
        bestFutureScore = float('-inf') 
        for futureState in futureStates:
            futurePos = futureState.getPacmanPosition()
            futureFood = futureState.getFood()
            futureGhostStates = futureState.getGhostStates()
            # Calculate the distance to the nearest food in the future state
            futureFoodDistances = [util.manhattanDistance(futurePos, food) for food in futureFood.asList()]
            nearestFutureFoodDistance = min(futureFoodDistances) if futureFoodDistances else 0

            # Calculate the distance to the nearest ghost in the future state
            futureGhostDistances = [util.manhattanDistance(futurePos, ghost.getPosition()) for ghost in futureGhostStates]
            nearestFutureGhostDistance = min(futureGhostDistances) if futureGhostDistances else 0

            # Check if the nearest ghost in the future state is scared
            nearestFutureGhostIndex = futureGhostDistances.index(nearestFutureGhostDistance)
            nearestFutureGhostScaredTime = futureGhostStates[nearestFutureGhostIndex].scaredTimer
            futureGhostScore = -nearestFutureFoodDistance if nearestFutureGhostScaredTime > 0 else nearestFutureGhostDistance

            # Calculate the future score
            futureScore = futureState.getScore() - nearestFutureGhostDistance - nearestFutureFoodDistance

            # If this future state is better than the best one seen so far, update the best score
            if futureScore > bestFutureScore:
                bestFutureScore = futureScore

        return successorGameState.getScore() - 0.7*nearestGhostDistance - 0.2*nearestFoodDistances - minSafeFoodDistance + 0.4*bestFutureScore

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        return self.minimaxSearch(gameState, agentIndex=0, depth=self.depth)[1]
    
    def minimaxSearch(self, gameState, agentIndex, depth):
        if depth == 0 or gameState.isLose() or gameState.isWin():

            return self.evaluationFunction(gameState), Directions.STOP
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        nextDepth = depth if nextAgent != 0 else depth - 1

        actions = gameState.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(gameState), None
        
        scores = [(self.minimaxSearch(gameState.generateSuccessor(agentIndex,action),nextAgent, nextDepth)[0], action)
                  for action in actions]
        if agentIndex == 0:  # Pacman
            return max(scores)
        else:  # a ghost
            return min(scores)

        """if agentIndex == 0:
            return max((self.minimaxSearch(gameState.generateSuccessor(agentIndex, action), nextAgent, nextDepth), action)
            for action in gameState.getLegalActions(agentIndex))
        else:  # a ghost
            return min((self.minimaxSearch(gameState.generateSuccessor(agentIndex, action), nextAgent, nextDepth), action)
                   for action in gameState.getLegalActions(agentIndex))
    def minimizer(self, gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(gameState), None
        nextAgent = agentIndex + 1
        if gameState.getNumAgents() == nextAgent:
            nextAgent = 0
            depth -= 1
        minScore = float("inf")
        minAction = None
        for action in actions:

            score = self.minimaxSearch(gameState.generateSuccessor(agentIndex, action), nextAgent, depth)[0]
            if score < minScore:
                minScore = score
                minAction = action
        return minScore, minAction
        
    def maximizer(self, gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(gameState), None
        nextAgent = agentIndex + 1
        if gameState.getNumAgents() == nextAgent:
            nextAgent = 0
            depth -= 1
        maxScore = float("-inf")
        maxAction = None
        for action in actions:
            score = self.minimaxSearch(gameState.generateSuccessor(agentIndex, action), nextAgent, depth)[0]
            if score > maxScore:
                maxScore = score
                maxAction = action
        return maxScore, maxAction

        util.raiseNotDefined()""" 
                

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        _, action = self.alphabeta(gameState, 0, self.depth, float('-inf'), float('inf'))
        return action
        """return self.alphabeta(gameState, 0, self.depth, float('-inf'), float('inf'))"""

        

    def alphabeta(self, gameState, agentIndex, depth, alpha, beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        if agentIndex == 0:  # Pacman (maximizing agent)
            return self.alphasearch(gameState, agentIndex, depth, alpha, beta)
        else:  # Ghosts (minimizing agents)
            return self.betasearch(gameState, agentIndex, depth, alpha, beta)

    def alphasearch(self, gameState, agentIndex, depth, alpha, beta):
        max_value = float('-inf')
        best_action = Directions.STOP
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        nextDepth = depth - 1 if nextAgent == 0 else depth

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            value, _ = self.alphabeta(successor, nextAgent, nextDepth, alpha, beta)
            if value > max_value:
                max_value, best_action = value, action
            if max_value > beta:
                return max_value, action
            alpha = max(alpha, max_value)
        return max_value, best_action

    def betasearch(self, gameState, agentIndex, depth, alpha, beta):
        min_value = float('inf')
        best_action = Directions.STOP
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        nextDepth = depth - 1 if nextAgent == 0 else depth

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            value, _ = self.alphabeta(successor, nextAgent, nextDepth, alpha, beta)
            if value < min_value:
                min_value, best_action = value, action
            if min_value < alpha:
                return min_value, action
            beta = min(beta, min_value)
        return min_value, best_action
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    INF = 100000.0
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        maxValue = -self.INF
        maxAction = Directions.STOP
        for action in gameState.getLegalActions(agentIndex=0):
            sucState = gameState.generateSuccessor(action=action, agentIndex=0)
            sucValue = self.expNode(sucState,currentDepth=0,agentIndex=1)
            if sucValue > maxValue:
                maxValue = sucValue
                maxAction = action
        return maxAction
    def maxNode(self, gameState, currentDepth):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        maxValue = -self.INF
        for action in gameState.getLegalActions(agentIndex=0):
            sucState = gameState.generateSuccessor(action = action, agentIndex = 0)
            sucValue = self.expNode(sucState,currentDepth,1)
            maxValue = max(maxValue,sucValue)
        return maxValue

    def expNode(self, gameState, currentDepth, agentIndex):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        nextAgent = agentIndex + 1
        if nextAgent >= gameState.getNumAgents():
            nextAgent = 0
            currentDepth += 1
        expValue = 0
        legalActions = gameState.getLegalActions(agentIndex)
        prob = 1.0/len(legalActions)
        for action in legalActions:
            sucState = gameState.generateSuccessor(action=action, agentIndex=agentIndex)
            if nextAgent == 0:
                sucValue = self.maxNode(sucState,currentDepth)
            else:
                sucValue = self.expNode(sucState, currentDepth, nextAgent)
            expValue += prob * sucValue
        return expValue


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Extract information from the current game state
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Calculate the distance to the nearest food
    foodDistances = [util.manhattanDistance(pos, foodPos) for foodPos in food.asList()]
    nearestFoodDistance = min(foodDistances) if foodDistances else 0

    # Calculate the distance to the nearest ghost
    ghostDistances = [util.manhattanDistance(pos, ghost.getPosition()) for ghost in ghostStates]
    nearestGhostDistance = min(ghostDistances) if ghostDistances else 0

    # Check if the nearest ghost is scared
    nearestGhostIndex = ghostDistances.index(nearestGhostDistance)
    nearestGhostScaredTime = scaredTimes[nearestGhostIndex]
    ghostScore = -nearestFoodDistance if nearestGhostScaredTime > 0 else nearestGhostDistance

    # Calculate the number of food and power pellets left
    numFoodLeft = currentGameState.getNumFood()
    numPowerPelletsLeft = len(currentGameState.getCapsules())

    # Calculate the total scared time of all ghosts
    totalScaredTime = sum(scaredTimes)

    # Combine the different factors to calculate the evaluation score
    score = currentGameState.getScore()
    score -= 0.5 * nearestGhostDistance
    score -= 1 * nearestFoodDistance
    score -= 1 * numFoodLeft
    score -= 1 * numPowerPelletsLeft
    score += 0.5 * totalScaredTime

    return score

# Abbreviation
better = betterEvaluationFunction
