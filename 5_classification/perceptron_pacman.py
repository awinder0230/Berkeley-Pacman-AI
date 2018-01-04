# perceptron_pacman.py
# --------------------
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


# Perceptron implementation for apprenticeship learning
import util
from perceptron import PerceptronClassifier
from pacman import GameState

PRINT = True


class PerceptronClassifierPacman(PerceptronClassifier):
    def __init__(self, legalLabels, maxIterations):
        PerceptronClassifier.__init__(self, legalLabels, maxIterations)
        self.weights = util.Counter()

    def classify(self, data ):
        """
        Data contains a list of (datum, legal moves)
        
        Datum is a Counter representing the features of each GameState.
        legalMoves is a list of legal moves for that GameState.
        """
        guesses = []
        for datum, legalMoves in data:
            vectors = util.Counter()
            for l in legalMoves:
                vectors[l] = self.weights * datum[l] #changed from datum to datum[l]
            guesses.append(vectors.argMax())
        return guesses


    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        ## Something you might need to use:
        ## trainingData[i][0][action]: feature vector of a state when taking an action ( f(s, a) )
        ## trainingData[i][1]: actions
        """
        self.features = trainingData[0][0]['Stop'].keys() # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
        print trainingData[0][0]
        print trainingData[0][1]

        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            for i in range(len(trainingData)):
                "*** YOUR CODE HERE ***"
                scores = util.Counter()
                for action in trainingData[i][1]:
                    scores[action] = self.weights * trainingData[i][0][action]
                # find the action with maximum score
                scorePairs = scores.items()
                scoreValue = [x[1] for x in scorePairs]
                maxValue = max(scoreValue)
                maxIndex = []
                for index, value in enumerate(scoreValue):
                    if value == maxValue:
                        maxIndex.append(index)
                predictAction = scorePairs[util.random.choice(maxIndex)][0]
                if predictAction != trainingLabels[i]:
                    self.weights = self.weights + trainingData[i][0][trainingLabels[i]] - trainingData[i][0][predictAction]