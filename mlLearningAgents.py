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

from pacman import Directions
from game import Agent
import random
import game
import util

# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 10):
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
        # Initialising previous state,action,value and score
        self.action = None
        self.state = None
        self.score = None
        self.q_val = dict()

    
    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

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

    
    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):

        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        print "Legal moves: ", legal
        print "Pacman position: ", state.getPacmanPosition()
        print "Ghost positions:" , state.getGhostPositions()
        print "Food locations: "
        print state.getFood()
        print "Score: ", state.getScore()
        print "type is ", type(self.state)

        # Getting the Pacman, Ghost and Food position
        pacs_position = state.getPacmanPosition()
        ghost_position = state.getGhostPositions()
        food_position = state.getFood()

        # Q value Initialisation
        if state not in self.q_val:
            self.q_val_initialisation(state, legal)
        
        # Q value Updatation
        if self.state != None:
            step = 'running'
            self.q_val_updation(state, step)

        # Updating the states and action
        self.update_states_and_action(state, legal)

        return self.action
            
    # q_val_initialisation
    #
    # Initliasing the Q values for all actions
    def q_val_initialisation(self, state, legal):
        self.q_val[state] = dict()
        for action in legal:
            if action not in self.q_val[state]:
                self.q_val[state][action] = 0.0

    # q_val_updation
    #
    # Updating the Q values using the reward and max q value
    def q_val_updation(self, state, step):
        
        # Get the Reward values
        reward = state.getScore() - self.score
        # Find the max q value for the state and action, if not in final state
        if step != 'Final':
            max_q_val = max(list(self.q_val[state].values()))
        else:
            max_q_val = 0.0

        # Update the q value as per the state and action
        self.q_val[self.state][self.action] += ( self.alpha * (reward + self.gamma * max_q_val - self.q_val[self.state][self.action]))

    # update_states_and_action
    #
    # Updating the state and action of the object using exploration and exploitation rates
    def update_states_and_action(self, state, legal):

        self.state = state
        self.score = state.getScore()

        random_probability = random.random()
        if random_probability < self.epsilon:
            self.action = random.choice(legal)
        else:
            q_action = None
            for action in legal:
                if q_action == None:
                    q_action = action
                if self.q_val[state][action] > self.q_val[state][q_action]:
                    self.action = action
                else:
                    self.action = q_action
        

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def reset_states_and_actions(self):
        self.action = None
        self.score = None
        self.state = None


    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):

        print "A game just ended!"
        
        if self.state != None:
            step = 'Final'
            self.q_val_updation(state,step)

        #Reseting the state and actions
        self.reset_states_and_actions()

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)


