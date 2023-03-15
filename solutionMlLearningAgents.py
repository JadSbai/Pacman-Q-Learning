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

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.1, gamma=0.8, numTraining=10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        # Stores the q values: associates a (state,action) pair to a q value.
        self.q_table = util.Counter()
        # Keeps track of the last score: used in the calculation of the reward at each state.
        self.previous_score = 0
        # Keeps track of the last recorded state.
        self.last_state = None
        # Keeps track of the last recorded action.
        self.last_action = None
        # Keeps track of the current reward.
        self.current_reward = 0

    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value

    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    # Returns the q-value associated to a (state,action) pair.
    def get_q_table_value(self, s, a):
        return self.q_table[(s, a)]

    # Updates the q-value associated to a (state,action) pair with value.
    def update_q_table_value(self, s, a, value):
        self.q_table[(s, a)] = value

    # Computes the reward r(s) for a given state s.
    # It equals the score at the current state minus the previous score.
    @staticmethod
    def compute_reward(previous_score, current_score):
        return current_score - previous_score

    # Returns the result of applying epsilon greedy strategy
    # Generates a random number r. If r < epsilon then Pacman will explore otherwise he will exploit.
    def epsilon_greedy_choice(self):
        return "Exploration" if random.random() <= self.epsilon else "Exploitation"

    # Computes the max q_value for a given state s.
    # Iterates through all states s' reachable from s.
    # Those correspond to the states Pacman can reach via the legal actions from s.
    # Returns the biggest q_value among those states.
    def get_max_q_value(self, state):
        legal_pacman_actions = state.getLegalPacmanActions()
        neighbouring_q_values = [self.get_q_table_value(state, action) for action in legal_pacman_actions]
        return max(neighbouring_q_values)

    # Computes the action a that maximises the q_value from a state s.
    # Like in get_q_value() method, iterates through all states s' reachable from s and returns the biggest q_value.
    # However, returns the action associated to that max q_value instead of the value itself.
    def get_action_max_q_value(self, state):
        legal_pacman_actions = state.getLegalPacmanActions()
        neighbouring_q_values = [(action, self.get_q_table_value(state, action)) for action in legal_pacman_actions]
        max_action_q_value_pair = max(neighbouring_q_values, key=lambda x: x[1])
        return max_action_q_value_pair[0]

    # Applies the q_value update equation.
    # current_q_value: q_value of the last state s, Q(s)
    # reward: reward of the last state, r(s)
    # alpha: learning rate
    # gamma: discount factor
    # max_q_value: maximum q value of current state s', Qmax(s')
    @staticmethod
    def apply_q_value_update_equation(current_q_value, alpha, reward, gamma, max_q_value):
        return current_q_value + alpha * (reward + gamma * max_q_value - current_q_value)

    # Carries out the q_value update step of Q-learning algorithm for the given state and action.
    def carry_out_q_value_update(self, state, action, max_q_value=0):
        current_q_value = self.get_q_table_value(state, action)
        new_q_value = self.apply_q_value_update_equation(current_q_value, self.alpha, self.current_reward, self.gamma,
                                                         max_q_value)
        self.update_q_table_value(state, action, new_q_value)

    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):

        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Gets the score of the current state.
        current_score = state.getScore()
        # Computes the reward obtained from the current state.
        self.current_reward = self.compute_reward(self.previous_score, current_score)
        exploration_action = random.choice(legal)
        # If it is the first move of the game, there is no previous state so far.
        # Hence, if it is the case, we do not update the Q table and go ahead with the e-greedy choice.
        # Otherwise, we carry out the update q-value step.
        if self.last_state is not None:
            max_q_value = self.get_max_q_value(state)
            self.carry_out_q_value_update(self.last_state, self.last_action, max_q_value)

        # Executes the epsilon-greedy strategy.
        if self.epsilon_greedy_choice() == "Exploration":
            # Exploration
            action_to_take = exploration_action
        else:
            # Exploitation
            action_to_take = self.get_action_max_q_value(state)

        # Updates intermediary variables.
        self.last_state = state
        self.last_action = action_to_take
        self.previous_score = current_score
        # We have to return an action
        return action_to_take

    # Handle the end of episodes
    # This is called by the game after a win or a loss.
    def final(self, state):
        # Updates Q(s,a) with (s,a) being the (state,action) pair associated to Pacman, when the game ended.
        current_score = state.getScore()
        self.current_reward = self.compute_reward(self.previous_score, current_score)
        self.carry_out_q_value_update(self.last_state, self.last_action)
        print("A game just ended!")

        # Resets intermediary variables for next game.
        self.last_state = None
        self.last_action = None
        self.previous_score = 0
        self.current_reward = 0
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        print("Episode", self.getEpisodesSoFar())
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
