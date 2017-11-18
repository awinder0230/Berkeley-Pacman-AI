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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        newPos  = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        currentFoodPos = currentGameState.getFood().asList()
        from searchAgents import mazeDistance
        newGhostPos  = [ghostState.getPosition() for ghostState in newGhostStates]
        newGhostDist = [mazeDistance((int(newPos[0]),int(newPos[1])), (int(ghostPos[0]),int(ghostPos[1])), successorGameState) for ghostPos in newGhostPos]
        newFoodPos   = newFood.asList()
        newFoodDist  = [mazeDistance((int(newPos[0]),int(newPos[1])), (int(foodPos[0]),int(foodPos[1])), successorGameState) for foodPos in newFoodPos]
        if successorGameState.isWin():
            return float('inf')
        else:
            minFoodDist  = min(newFoodDist)
            minGhostDist = min(newGhostDist)
            if minGhostDist <= 1:
                return 0.0
            elif newPos in currentFoodPos:
                return 2.0
            else:
                return 1.0 / minFoodDist # float

            '''
            minFoodDist = min(newFoodDist)
            minGhostDist = min(newGhostDist)
            if minGhostDist <= 2 or newPos in newGhostPos:
                print("danger")
                return 0
            elif newPos in currentFoodPos:
                print("eat food")
                return 99999
            else: 
                print("score: " + str(float(1000)/minFoodDist))
                return float(1000)/minFoodDist
            '''


        # return successorGameState.getScore() default return value

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
        """
        "*** YOUR CODE HERE ***"

        def minimax(newState, depth, index, numGhosts):
            
            # terminal test
            if (depth == 0) or newState.isLose() or newState.isWin():
                return self.evaluationFunction(newState)

            legalMoves = newState.getLegalActions(index)
            nextStates = [newState.generateSuccessor(index,move) for move in legalMoves]

            if index == 0: # pacman
                if numGhosts == 0: # no ghosts
                    return max([minimax(state, depth - 1, index, numGhosts) for state in nextStates])
                else:
                    return max([minimax(state, depth, index + 1, numGhosts) for state in nextStates])
            else: # ghost moves
                if index == numGhosts: # last ghost, next index should be 0 for pacman's turn
                    return min([minimax(state, depth - 1, 0, numGhosts) for state in nextStates])
                else:
                    return min([minimax(state, depth, index + 1, numGhosts) for state in nextStates])
        
        # debugging message
        #score = minimax(gameState, self.depth, 0, gameState.getNumAgents()-1)
        #print("score: %d" % score)

        numGhosts = gameState.getNumAgents() - 1
        legalActions = gameState.getLegalActions()
        nextGameStates = [gameState.generateSuccessor(0,action) for action in legalActions]

        if numGhosts == 0:
            scores = [minimax(nextGameState, self.depth - 1, 0, 0) for nextGameState in nextGameStates]
        else:
        	scores = [minimax(nextGameState, self.depth, 1, numGhosts) for nextGameState in nextGameStates]

        bestScore = max(scores)
        bestIndices = [i for i in range(len(scores)) if scores[i] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return legalActions[chosenIndex]



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphabeta(newState, depth, index, numGhosts, alpha, beta):
            
            # terminal test
            if (depth == 0) or newState.isLose() or newState.isWin():
                return self.evaluationFunction(newState)

            legalMoves = newState.getLegalActions(index)
            nextStates = [newState.generateSuccessor(index,move) for move in legalMoves]

            if index == 0: # pacman
                if numGhosts == 0: # no ghosts
                    newDepth = depth - 1
                    newIndex = index
                else:
                    newDepth = depth
                    newIndex = index + 1

                returnValue = -float('inf')
                for state in nextStates:
                    returnValue = max(returnValue, alphabeta(state, newDepth, newIndex, numGhosts, alpha, beta))
                    if returnValue >= beta:
                        return returnValue
                    alpha = max(alpha, returnValue)
                return returnValue
            else: # ghost moves
                if index == numGhosts: # last ghost, next index should be 0 for pacman's turn
                    newDepth = depth - 1
                    newIndex = 0
                else:
                    newDepth = depth
                    newIndex = index + 1
                returnValue = float('inf')
                for state in nextStates:
                    returnValue = min(returnValue, alphabeta(state, newDepth, newIndex, numGhosts, alpha, beta))
                    if returnValue <= alpha:
                        return returnValue
                    beta = min(beta, returnValue)
                return returnValue
        
        # debugging message
        #score = alphabeta(gameState, self.depth, 0, gameState.getNumAgents()-1, -float('inf'), float('inf'))
        #print("score: %d" % score)

        numGhosts = gameState.getNumAgents() - 1
        legalActions = gameState.getLegalActions()
        nextGameStates = [gameState.generateSuccessor(0,action) for action in legalActions]

        if numGhosts == 0: # no ghosts
            newDepth = self.depth - 1
            newIndex = 0
        else:
            newDepth = self.depth
            newIndex = 1

        returnVal = -float('inf')
        alpha     = -float('inf')
        beta      =  float('inf')
        bestIndex = -1
        for index, nextGameState in enumerate(nextGameStates):
            tmpVal = alphabeta(nextGameState, newDepth, newIndex, numGhosts, alpha, beta)    
            if returnVal < tmpVal:
        	    returnVal = tmpVal
        	    bestIndex = index
            if returnVal >= beta:
                break
            alpha = max(alpha, returnVal)

        return legalActions[bestIndex]

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
        def expectiMax(newState, depth, index, numGhosts):
            
            # terminal test
            if (depth == 0) or newState.isLose() or newState.isWin():
                return self.evaluationFunction(newState)

            legalMoves = newState.getLegalActions(index)
            nextStates = [newState.generateSuccessor(index,move) for move in legalMoves]

            if index == 0: # pacman
                if numGhosts == 0: # no ghosts
                    return max([expectiMax(state, depth - 1, index, numGhosts) for state in nextStates])
                else:
                    return max([expectiMax(state, depth, index + 1, numGhosts) for state in nextStates])
            else: # ghost moves
                if index == numGhosts: # last ghost, next index should be 0 for pacman's turn
                    newDepth = depth - 1
                    newIndex = 0
                else:
                    newDepth = depth
                    newIndex = index + 1

                totalSum = 0.0 # float
                for state in nextStates:
                    totalSum += expectiMax(state, newDepth, newIndex, numGhosts)
                return totalSum / len(nextStates)

        numGhosts = gameState.getNumAgents() - 1
        legalActions = gameState.getLegalActions()
        nextGameStates = [gameState.generateSuccessor(0,action) for action in legalActions]

        if numGhosts == 0:
            scores = [expectiMax(nextGameState, self.depth - 1, 0, 0) for nextGameState in nextGameStates]
        else:
        	scores = [expectiMax(nextGameState, self.depth, 1, numGhosts) for nextGameState in nextGameStates]

        bestScore = max(scores)
        bestIndices = [i for i in range(len(scores)) if scores[i] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalActions[chosenIndex]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>

      The basic idea of the following implementation is that
      eating ghost is the first priority. This idea is based
      on the observation that to exceed 1450 points in average,
      it requires eating at least 3 or 4 ghosts in one game. 

      Note that the following strategy only works for smallClassic
      layout, but the basic idea could be apply to many other layouts

      In order to win the game with score higher than 1700, the pacman
      could follow the following steps:

          1. After the game starts, eat the closest capsule
          2. After eating capsule, eat as many ghosts as possible
          3. After eating all the ghosts or there is no enough time 
             to eat the next closest ghost, repeat step 1 and step 2
             until there is no more capsules.
             (in smallClassic case, there are only 2 capsules in total)
          4. Clean up all the food in the map

      Thus, to design an evaluation function especially for smallClassic
      layout, here I designed 5 cases besides the winning and losing gateState:

          case 1: before eating the 1st capsule
          case 2: after eating the 1st capsule and there are scared ghosts
          case 3: after eating the 1st capsule but there are no more scared ghosts
          case 4: after eating the 2nd capsule and there are scared ghosts
          case 5: there is no more any capsule and scared ghost in the map

      The following description would explain the design of return value
      and other details of each case:

          case 0:

              Before start, first check whether the gameState is the end of
              the game, return +99999 for winning case and -99999 for losing case.

              Also, if pacman is only 1 step beside any not scared ghost, return
              -10000 for this case since this may result in losing case.

          case 1:

              1st priority: eat the closest capsule

              to achieve this goal, return value is designed to be:

                  200 - len(capsules) * 100 + 1.0 / min(capsulesDist)

              where len(capsules) is number of capsules and min(capsulesDist)
              is the distance to the nearest capsule.

              This return value encourage pacman to go after capsule so as
              to decrease the value of min(capsulesDist).

              return value min: 200 - 2 * 100 + 0.xx  = 0.xx
              return value max: 200 - 2 * 100 + 1 / 1 = 1 (1 step before eating 1st capsule)

          case 2:

              subcase 2-1: 

              1st priority: eat the closest ghost

              to achieve this goal, return value is designed to be:

                  401 - scaredNum * 100 + 1.0 / pair[0]

              where scaredNum is number of scared ghosts, and pair[0] is 
              the distance to the nearest scared ghost.

              This return value encourage pacman to go after ghosts so as
              to decrease the value of pair[0]. Despite the value of pair[0]
              may increase right after eating the first ghost, scaredNum would
              decrease by 1 which in turn getting a much higher return value.

              return value min: 401 - 2 * 100 + 0.xx = 201.xx
              return value max: 401 - 1 * 100 + 1/1  = 302 (1 step before eating 2nd ghost)

              Note the the minimum value in this case should be larger than the
              maximum value in case 1, so that the pacman would be encouraged to 
              go from case 1 to case 2.

              subcase 2-2: 

              Lat but not least, if the remaining scared time is not enough to 
              chase any remaining scared ghost, then the 1 priority for pacman
              is to eat the 2nd capsule instead of keep chasing ghosts. Thus the
              return value for this scenario is designed to be:

                  300 - len(capsules) * 100 + 1.0 / min(capsulesDist)

              return value min: 300 - 1 * 100 + 0.xx = 200.xx
              return value max: 300 - 1 * 100 + 1/1  = 201 (1 step before eating 2nd capsule)

              Note that the maximum value in this subcase should be less than any value
              in subcase 2-1, since we should encourage pacman to eat all the scared ghosts
              as fast as possible. Also, any value in this subcase should be larger than
              any value in case 1, too.

          case 3:

              1st priority: eat the 2nd capsule

              to achieve this goal, return value is designed to be:

                  500 - len(capsules) * 100 + 1.0 / min(capsulesDist)

              return value min: 500 - 1 * 100 + 0.xx = 400.xx
              return value max: 500 - 1 * 100 + 1/1  = 401

              Note that the minimum value for this case should be larger
              than any value in case 2 to encourage pacman eating all the ghosts

          case 4:

              subcase 4-1:

              1st priority: eat the closest scared ghost

              to achieve this goal, return value is designed to be:

                  801 - scaredNum * 100 + 1.0 / pair[0]

              which have the same idea in case 2.

              return value min: 801 - 2 * 100 + 0.xx = 601.xx
              return value max: 801 - 1 * 100 + 1/1  = 702

              Note that the minimum return value have to be larger than
              any value in case 3 to encourage pacman going from case 3
              to case 4.

              subcase 4-2:

              Same as subcase 2-2. if there is no more time left for eating
              the next closest ghost, start cleaning up the map instead of
              keep chasing ghosts. To achieve this goal, return value is
              designed to be:

                  600 - foodNum + 1.0 / minFoodDist

              where foodNum is number of remaining food, and minFoodDist is
              the distance to the nearest food.

              This return value encourage pacman to go after the nearest food
              as well as eating all the remaining food.

              return value min: 600 - foodNum + 0.xx
              return value max: 600 - 1 + 1/1 = 600 (1 step before eating last food)

              Note that 0 < foodNum < 100, thus the minimum of return value is larger
              than 500 but less than 600, which matches the requirement that return value
              in this case should be less than subcase 4-1 but larger than any value in
              case 3.

              0 < foodNum  : if foodNum = 0 implies winning case
              foodNum < 100: there is no such space for 100 foodNum in this layout

          case 5:

              1st priority: eat the nearest food

              return value:

                  1000 - foodNum + 1.0 / minFoodDist

              return value min: 900.xx (0 < foodNum < 100)
              return value max: 1000 - 1 + 1/1 = 1000

              Note that the minimum value in this case should be larger than
              any other value in any other case described above.
    """
    "*** YOUR CODE HERE ***"
    # pacman position
    pacmanPos_tmp = currentGameState.getPacmanPosition()
    pacmanPos = (int(pacmanPos_tmp[0]), int(pacmanPos_tmp[1]))

    # capsule positions
    capsules = currentGameState.getCapsules()

    # scared timer for ghosts
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [state.scaredTimer for state in ghostStates]

    # import mazeDistance, remember to include related files
    from searchAgents import mazeDistance

    # count the distance between pacman and each ghost
    ghostsPos_tmp = currentGameState.getGhostPositions()
    ghostsPos = [(int(tmp[0]), int(tmp[1])) for tmp in ghostsPos_tmp]
    ghostsDist = [mazeDistance(ghostPos, pacmanPos, currentGameState) for ghostPos in ghostsPos]

    # case 0
    if currentGameState.isWin():
        return 99999
    elif currentGameState.isLose():
        return -99999

    # build pairs between scared ghosts and their remaining scared time
    # count the number of scared ghost at the same time
    distTimerPairs = []
    effectGhostDist = []
    scaredNum = 0
    for i in range(len(ghostStates)):
        distTimerPairs.append((ghostsDist[i], scaredTimers[i]))
        if scaredTimers[i] == 0:
            effectGhostDist.append(ghostsDist[i])
        else:
            scaredNum += 1

    # case 0
    if len(effectGhostDist) != 0:
        if min(effectGhostDist) <= 1:
            return -10000

    # sort according to distance of scared ghosts
    distTimerPairs.sort()

    # distance to capsules
    capsulesDist = [mazeDistance((int(capsule[0]),int(capsule[1])), pacmanPos, currentGameState) for capsule in capsules]

    if len(capsules) == 2: # case 1
        # eat capsules is 1st priority
        # 1 step before eating 1st capsule, return value is 200 - 2 * 100 + 1 = 1
        return 200 - len(capsules) * 100 + 1.0 / min(capsulesDist)
    elif len(capsules) == 1 and scaredNum != 0: # case 2
        # subcase 2-1
        # eat ghosts is 1st priority
        # return value for 1 step aftering eating 1st capsule have to be larger than 1 (201.x > 1)
        for pair in distTimerPairs:
            if (pair[1] - pair[0] >= 0) and (pair[1] > 0):
                # 1 step before eating 2nd ghost is 401 - 100 + 1 = 302
            	return 401 - scaredNum * 100 + 1.0 / pair[0]
            else:
                continue
        # subcase 2-2
        # if script end up in this section, meaning there is no more time for eating next ghost
        # then the 1st priority will be eating 2nd capsule
        # return value have to be less than any value of upper section (more prefer to eat ghost rather than wasting capsules)
        # 1 step before eating 2nd capsule is 300 - 100 + 1 = 201 (201.x > 201)
        return 300 - len(capsules) * 100 + 1.0 / min(capsulesDist)
    elif len(capsules) == 1 and scaredNum == 0: # case 3
        # after eating 2 ghosts using 1st capsule
        # eating 2nd capsule is 1st priority
        # minimum value have to be larger than the value of 1 step before eating 2nd ghost (302)
        # so as to encourage eating 2nd ghost
        # 1 step after eating 2nd ghost is 400.x
        # 1 step before eating 2nd capsule is 500 - 100 + 1 = 401
        return 500 - len(capsules) * 100 + 1.0 / min(capsulesDist)
    elif len(capsules) == 0 and scaredNum != 0: # case 4
        # subcase 4-1
        # eat ghost is 1st priority
        # return value for 1 step aftering eating 2nd capsule have to be larger than 401 ( 601.x > 401)
        for pair in distTimerPairs:
            if (pair[1] - pair[0] >= 0) and (pair[1] > 0):
                # 1 step before eating 2nd ghost is 801 - 100 + 1 = 702
            	return 801 - scaredNum * 100 + 1.0 / pair[0]
            else:
                continue
        # subcase 4-2
        # if script end up in this section, meaning there is no more time for eating next ghost
        # then the 1st priority will be eating the nearset food
        # return value have to be larger than 401 (1 step before eating 2nd capsule)
        # return value have to be less than 601.x (any value of upper section)
        foodList = currentGameState.getFood().asList()
        foodDist = [mazeDistance((int(foodPos[0]), int(foodPos[1])), pacmanPos, currentGameState) for foodPos in foodList]
        minFoodDist = min(foodDist)
        foodNum  = len(foodList) 
        return 600 - foodNum + 1.0 / minFoodDist
    else:
    	# eating the nearest food
    	# code is same as upper section, but the meaning is different
    	# here the minimum value have to be larger than any other value above
    	foodList = currentGameState.getFood().asList()
        foodDist = [mazeDistance((int(foodPos[0]), int(foodPos[1])), pacmanPos, currentGameState) for foodPos in foodList]
        minFoodDist = min(foodDist)
        foodNum  = len(foodList) 
        return 1000 - foodNum + 1.0 / minFoodDist
    
# Abbreviation
better = betterEvaluationFunction

