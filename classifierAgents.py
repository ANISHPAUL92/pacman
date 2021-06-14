# classifierAgents.py
# parsons/07-oct-2017
#
# Version 1.0
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
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

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split

# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print "Initialising"

    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.
    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray
                
    # This gets run on startup. Has access to state information.
    #
    # Here we use it to load the training data.
    def registerInitialState(self, state):

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])

        # data and target are both arrays of arbitrary length.
        #
        # data is an array of arrays of integers (0 or 1) indicating state.
        #
        # target is an array of imtegers 0-3 indicating the action
        # taken in that state.
                
    def getPriorProbability(self, train_class):
        # Initialising the prior probability
        prior = np.empty(4)

        # Assigning count the number of times each move occurs
        move,counts = np.unique(train_class, return_counts=True)

        # Calculating the prior probability, using float because the values will be decimals
        prior = counts / float(len(train_class))

        return prior, counts        


    def getLikelihoodProbability(self, train_val, train_class, counts):
        # Initialising the Likelihood
        Likelihood = np.empty([4, 25])
        train_data = np.concatenate((train_val,train_class), axis=1)
        for i in range(4):
            for j in range(25):
                # Whenever the class value is same as i, we add it to data  
                data = train_data[train_data[:, -1] == i]
                # Below we sum each column of the list present in data and divide by the count value of each class
                likelihood = float(data[:, j].sum()) / float(counts[i])
                Likelihood[i, j] = likelihood
        
        return Likelihood

    def getActionValueUsingPosterior(self, feature_values, prior):
        move_option = np.ones(4)
        posterior = np.empty(4)
        # Computing the posterior probability for each move as per the test data
        for i in range(4):
            for j in range(len(feature_values[0])):
                move_option[i] = move_option[i] * feature_values[i, j]
            posterior[i] = move_option[i] * prior[i]

        # Returning the move with the maximum probability
        return np.argmax(posterior)

    # Tidy up when Pacman dies
    def final(self, state):

        print "I'm done!"

    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do
    def getAction(self, state):

        # How we access the features.
        features = api.getFeatureVector(state)

        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=0.20)
        # Modifying the input bits from data to an array of inputs for the classifier
        X_train = np.array(X_train)  
        # Reshaping the class values from a list to an array of output values
        y_train = np.array(y_train).reshape(-1, 1)
        
        # Returns prior probability and the count for each move
        prior_prob, count_val = self.getPriorProbability(y_train)
        
        # Returns likelihood probability of each bit for each move
        likelihood_prob = self.getLikelihoodProbability(X_train, y_train, count_val)

        feature_vals_likelihood = np.empty([4,25])
        # Assigning liklihood probability to test data
        for i in range(len(features)):
            feature_vals_likelihood[:, i] = features[i] * likelihood_prob[:, i ]    
        feature_vals_likelihood[feature_vals_likelihood ==0 ] = 1
        
        actionval = self.getActionValueUsingPosterior(feature_vals_likelihood,prior_prob)

        action = self.convertNumberToMove(actionval)
        
        # Get the actions we can try.
        legal = api.legalActions(state)
        #Check if the action is in legal and make move or else move randomly
        if action in legal:
            return api.makeMove(action, legal)
        else:
            return api.makeMove(random.choice(legal), legal)
