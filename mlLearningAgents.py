# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util
from functools import reduce


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        # Get the current legal actions and remove the STOP action
        legalActions = state.getLegalPacmanActions()
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)

        self.legalActions = legalActions
        self.state = state

    def getLegalActions(self):
        return self.legalActions

    def __eq__(self, other):
        """
        Allows two states to be compared.
        """
        return self.state.data == other.state.data

    def __hash__(self):
        """
        Allows states to be keys of dictionaries.
        """
        return hash(self.state.data)


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.1,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyper-parameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        self.qDict = util.Counter()
        self.visitationCount = util.Counter()
        self.previousState = None
        self.previousAction = None
        self.episodesSoFar = 0

    # Getter functions for the parameters
    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # Setter functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory.
            The game score of the resulting state minus the game score at the starting state.
        """
        return endState.getScore() - startState.getScore()

    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            The Q value of the corresponding state-action pair.
        """
        return self.qDict[(state, action)]

    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        legalActions = state.getLegalActions()

        # Map each action to its given q-value based on the current state
        qValues = map(lambda action: self.getQValue(state, action), legalActions)
        qValuesList = list(qValues)  # Convert to a list
        length = len(qValuesList)

        if length > 0:
            return max(qValuesList)
        else:
            return 0

    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was taken
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        currentQValue = self.getQValue(state, action)
        updatedQValue = currentQValue + self.alpha * (reward + self.gamma * self.maxQValue(nextState) - currentQValue)
        self.qDict[(state, action)] = updatedQValue

    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Increment the stored visitation count for the given state-action pair by 1.

        Args:
            state: Starting state
            action: Action taken
        """
        self.visitationCount[(state, action)] = self.getCount(state, action) + 1

    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        return self.visitationCount[(state, action)]

    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes the exploration function. Takes into account the utility of the state-action pair
        as well as its visitation count (improved version of a least-pick)

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: How many times the given state-action pair has been visited so far.

        Returns:
            The exploration value
        """
        if counts == 0:
            return abs(utility)
        else:
            if utility > 0:
                # The proportion of utility to counts prioritizes unvisited cells, or high utility ones.
                return abs(utility / counts)
            else:
                # Prioritise exploring cells that have negative utility for improved learning.
                return abs(utility ** 8 / counts)

    def getBestExplorationAction(self, state: GameStateFeatures) -> Directions:
        """
        Args: state: Starting state Returns: The best action (i.e, the one with the greatest exploration value) to
        take in a given state following our exploration criteria.
        """
        bestAction = Directions.STOP  # Default action
        maxExplorationValue = float('-inf')
        for action in state.getLegalActions():
            qValue = self.getQValue(state, action)  # The utility of the state-action-pair
            count = self.getCount(state, action)  # The visitation count of the state-action-pair
            explorationValue = self.explorationFn(qValue, count)
            if explorationValue > maxExplorationValue:
                bestAction = action
                maxExplorationValue = explorationValue
        return bestAction

    def getBestExploitationAction(self, state: GameStateFeatures) -> Directions:
        """
        Selects the action with the highest Q-Value out of all legal actions.
        Args:
            state: Starting state

        Returns:
            Best action
        """
        legalActions = state.getLegalActions()
        return reduce(lambda action, bestAction: action if self.getQValue(state, action) > self.getQValue(state, bestAction) else bestAction, legalActions)

    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """

        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        stateFeatures = GameStateFeatures(state)

        if self.previousState:
            # Learn, based on current reward value and information from previous state.
            current_reward = self.computeReward(self.previousState, state)
            previousStateFeatures = GameStateFeatures(self.previousState)
            self.learn(previousStateFeatures, self.previousAction, current_reward, stateFeatures)

        if random.random() <= self.epsilon:
            # Based on max ExplorationFn value, returns explorative action to take
            chosenAction = self.getBestExplorationAction(stateFeatures)
        else:
            # Based on max Q value, returns exploitative action to take
            chosenAction = self.getBestExploitationAction(stateFeatures)

        self.previousState = state
        self.previousAction = chosenAction
        self.updateCount(stateFeatures, chosenAction)
        return chosenAction

    def final(self, state: GameState):
        """
        Handle the end of a game. Resets class values and runs a final version of learn().

        Running learn again is essential because we need to consider the real rewards associated to loosing or winning.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")
        stateFeatures = GameStateFeatures(state)
        previousStateFeatures = GameStateFeatures(self.previousState)

        finalReward = self.computeReward(self.previousState, state)
        self.learn(previousStateFeatures, self.previousAction, finalReward, stateFeatures)

        # Reset intermediary variables before running the next game.
        self.previousState = None
        self.previousAction = None

        """
        Keep track of the number of games played, and set learning parameters
        to zero when we are done with the pre-set number of training episodes
        """
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
