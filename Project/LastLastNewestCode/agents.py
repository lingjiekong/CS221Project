'''
Created on Dec 1, 2016
@author: Stanley Jacob
'''
import gym
import logging
import numpy as np
import math, time
import tensorflow as tf
import pickle
from gym import spaces
import os.path
import warnings
warnings.filterwarnings("ignore")


########################################################################
# use for mlp
def inputActivationFunction(activationFlag, currentLayer, weights, biases):
    if activationFlag == 1:
        nextLayer = tf.nn.tanh(tf.add(tf.matmul(currentLayer, weights), biases))
    elif activationFlag == 2:
        nextLayer = tf.nn.relu(tf.add(tf.matmul(currentLayer, weights), biases))
    elif activationFlag == 3:
        nextLayer = tf.nn.sigmoid(tf.add(tf.matmul(currentLayer, weights), biases))
    elif activationFlag == 4:
        nextLayer = tf.nn.elu(tf.add(tf.matmul(currentLayer, weights), biases))
    return nextLayer

def outputCalc(currentLayer, outputFlag, weights, biases):
    if outputFlag == 1:
        outputLayer = tf.matmul(currentLayer, weights) + biases
    elif outputFlag == 2:
        outputLayer = tf.sigmoid(currentLayer, weights) + biases
    return outputLayer

def regularizationUpdate(regulTerm, weights, totalLayers, regParam, regFlag):
    if regFlag == 1:
        regulTerm += tf.nn.l2_loss(weights) * regParam[totalLayers - 2] 
        return regulTerm

def getLearningRate(startLearnRate, globalStep, decayStep, decayRate, staircase, learnFlag):
    if learnFlag == 1:
        return tf.train.exponential_decay(startLearnRate, globalStep, decayStep, \
                                                    decayRate, staircase=True)

def determineWeights(currentLayer, n, n_hidden, regParam, regulTerm, activationFlag):
    for i in range(1, n - 1):
        weights = tf.Variable(
            tf.truncated_normal(                \
                [n_hidden[i - 1], n_hidden[i]], \
                stddev=1. / math.sqrt(float(n_hidden[i - 1]))))
        biases = tf.Variable(tf.zeros([n_hidden[i]]))
        nextLayer = inputActivationFunction(activationFlag, currentLayer, weights, biases)
        currentLayer = nextLayer
        regulTerm += tf.nn.l2_loss(weights) * regParam[i - 1]
    return (weights, biases, currentLayer, regulTerm)

def chooseOptimizer(decayRate, momentum, cost, global_step, calcLearn, optimizerChoice):
    if optimizerChoice == 1:
        return tf.train.RMSPropOptimizer(calcLearn, decayRate, momentum,  \
            ).minimize(cost,         \
            global_step)
    else:
        return tf.train.GradientDescentOptimizer(calcLearn, \
            use_locking=False, name='GradientDescent').minimize(cost, \
            global_step)
# end of use for mlp
########################################################################


## Multilayer perceptron algorithm implementation
## Uses tensor flow 
########################################################################
# use for initCostOpt
def mlp(inputLayer, n_hidden, regParam, activationFlag, outputFlag, regFlag):

    currentLayer = inputLayer
    totalLayers = len(n_hidden)
    regulTerm = 0.

    (weights, biases, currentLayer, regulTerm) = \
        determineWeights(currentLayer, totalLayers, n_hidden, regParam, regulTerm, activationFlag)

    weights = tf.Variable(tf.truncated_normal([n_hidden[totalLayers - 2], \
                    n_hidden[totalLayers - 1]], \
                    stddev=1. / math.sqrt(float(n_hidden[totalLayers - 2]))))
    biases = tf.Variable(tf.zeros([n_hidden[totalLayers - 1]]))

    outputLayer = outputCalc(currentLayer, outputFlag, weights, biases)
    regulTerm = regularizationUpdate(regulTerm, weights, totalLayers, regParam, regFlag)

    return outputLayer, regulTerm
# end of use for initCostOpt
########################################################################

class deepQAgent(object):
    def __del__(self):
        self.close()

    def __init__(self, observation_space, action_space, reward_range):
        tf.set_random_seed(1)
        np.random.seed(1)

        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_range = reward_range

        self.numStates = self.observation_space.shape[0]
        self.numActions = self.action_space.n
        self.x = tf.placeholder("float", [None, self.numStates])
        self.y = tf.placeholder("float", [None, 1])
        self.currentAction = tf.placeholder("float", [None, self.numActions])

        self.settingsInits()
        self.tunedNNInits()
        self.initCostOpt()

    ########################################################################
    # use for __init__
    def settingsInits(self):
        self.discountRate = 0.99

        self.startLearnRate = 0.008
        self.learnDecay = 0.999

        self.epsilonProb = 0.15
        self.epsilonDecay = 0.996

        self.lambdaProb = 0.1
        self.updateProb = 0.25

        self.batchSize = 75
        self.pastActionsSize = 50000
        self.pastActions = []

    def tunedNNInits(self):
        self.activationFlag = 1
        self.outputFlag = 1
        self.regFlag = 1
        self.hiddenNum = 300
        self.regulParam = [0.0001, 0.000001]

        self.global_step = tf.Variable(0, trainable=False)
        decayStep = 100
        staircase = True
        self.learnFlag = 1
        self.calcLearn = getLearningRate(self.startLearnRate, self.global_step, \
            decayStep, self.learnDecay, staircase, self.learnFlag)

    def initCostOpt(self):
        self.utility, regTerm = mlp(self.x, [self.numStates] + [self.hiddenNum] + [self.numActions],
                                              self.regulParam, self.activationFlag, \
                                              self.outputFlag, self.regFlag)
        self.someUtility = tf.reduce_sum(self.currentAction * self.utility, reduction_indices=1)  
        self.someUtility = tf.reshape(self.someUtility, [-1, 1])

        self.lossFull = pow((self.someUtility - self.y), 2)
        self.cost = regTerm + tf.reduce_mean(self.lossFull)

        decayRate = 0.9
        momentum = 0.1

        self.optimizerChoice = 1
        self.optimizer = chooseOptimizer(decayRate, momentum, self.cost, \
            self.global_step, self.calcLearn, self.optimizerChoice)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.sess.run(tf.assign(self.global_step, 0))
    # end of use for __init__
    ########################################################################

    ########################################################################
    # use for run.py    
    def scaleStates(self, someState):
        if np.isinf(self.observation_space.low).any() or np.isinf(self.observation_space.high).any():
            return someState
        else:
            scaledState = 3 * ((someState - self.observation_space.low) / (
                self.observation_space.high - self.observation_space.low) * 2. - 1.)
            return scaledState

    def getEpsilon(self, trial):
        if trial != None:
            return self.epsilonProb * pow(self.epsilonDecay, trial)
        else:
            return 0.
    # end of use for run.py
    ########################################################################
    
    def setEpsilon(self, newEpsilonValue):
        self.epsilonProb = newEpsilonValue

    ########################################################################
    # use for getNewUtil
    def iteratePossible(self, possUtil, newState, differ, dR, maxVal, index):
        while newState != None and differ < maxVal:
                differ += newState
                newIndex = index + differ

                possUtil += dR * self.pastActions[newIndex][2]
                dR = dR * self.discountRate
                if self.pastActions[newIndex][6] == None or not (differ < maxVal):
                    possUtil += dR * self.value(self.pastActions[newIndex][3]) * self.pastActions[newIndex][4]
                newState = self.pastActions[newIndex][6]
        return possUtil
    # end of use for getNewUtil
    ########################################################################
    

    ########################################################################
    # use for update updateMemory
    def getNewUtil(self, index, newState, reward, someState, rate):
        if self.lambdaProb > 0.:
            maxVal = 1000
            possUtil = reward
            dR = self.discountRate
            differ = 0
            if newState == None:
                possUtil += dR * self.value(someState) * rate    
            possUtil = self.iteratePossible(possUtil, newState, differ, dR, maxVal, index)
        else:
            possUtil = 0.
        self.pastActions[index][5] = possUtil
        return possUtil
    # end of use for updateMemory
    ########################################################################


    ########################################################################
    # use for update runMemoryUpdate
    def updateMemory(self, index):
        state, action, reward, someState, rate, utility, newState = self.pastActions[index]

        if utility != None:
            possUtil = utility
        else:
            possUtil = self.getNewUtil(index, newState, reward, someState, rate)
        
        possUtil = possUtil * self.lambdaProb + (reward + \
            self.discountRate * self.value(someState) * rate) * (1. - self.lambdaProb)

        return possUtil, state, action

    # end of use for runMemoryUpdate
    ########################################################################
    
    ########################################################################
    # use for update learn
    def runMemoryUpdate(self, utilList, stateList, actionList):
        if len(self.pastActions) > self.batchSize:
            searchSize = np.random.choice(len(self.pastActions), self.batchSize) 
            for index in searchSize:
                someUtil, someState, someAction = self.updateMemory(index)
                utilList = np.concatenate((utilList, someUtil), 0)  
                stateList = np.concatenate((stateList, someState.reshape(1, -1)), 0)
                actionList = np.concatenate((actionList, np.array([someAction])), 0)
                
            fullActions = np.zeros((stateList.shape[0], self.numActions))
            fullActions[np.arange(actionList.shape[0]), actionList] = 1.

            self.sess.run(self.optimizer, feed_dict={self.x: stateList, self.y: utilList.reshape((-1, 1)),
                                                     self.currentAction: fullActions})
    # end of use use for update learn
    ########################################################################

    ########################################################################
    # use for run.py
    def learn(self, state, action, newState, reward, finishFlag, nextaction):
        util = reward + self.discountRate * self.value(newState) * finishFlag
        
        util = util.reshape(1, )
        stateList = state.reshape(1, -1)
        actionList = np.array([action])
        utilList = util
        
        if (np.random.random() < self.updateProb):
            self.runMemoryUpdate(utilList, stateList, actionList)

        fullActions = np.zeros((stateList.shape[0], self.numActions))
        fullActions[np.arange(actionList.shape[0]), actionList] = 1.
        self.sess.run(self.lossFull, feed_dict={self.x: stateList, self.y: utilList.reshape((-1, 1)),
                                                          self.currentAction: fullActions})
        
        if len(self.pastActions) > 0 and np.array_equal(self.pastActions[-1][3], state):
            self.pastActions[-1][6] = 1
        self.pastActions.append([state, action, reward, newState, finishFlag, None, None])
        self.pastActions = self.pastActions[-self.pastActionsSize:]
    # end of use for run.py
    ########################################################################

    def value(self, consideredState):
        if consideredState.ndim == 1:
            consideredState = consideredState.reshape(1, -1)
        return np.max(self.sess.run(self.utility, feed_dict={self.x: consideredState})).reshape(1, )

    def getBestAction(self, consideredState):
        if consideredState.ndim == 1:
            consideredState = consideredState.reshape(1, -1)
        return np.argmax(self.sess.run(self.utility, feed_dict={self.x: consideredState}))

    def controlsCalc(self, environ, state):
        # possible actions
        g_noEngineFired = 0
        g_leftEngineFired = 1
        g_bottomEngineFired = 2
        g_rightEngineFired = 3

        horPosWeight = 0.5
        horSpeedWeight = 1.
        angleThresh = 0.4
        targetVertWeight = 0.2
        angleMax = 0.05
        vertMax = 0.05
        newVertWeight = 0.5
        newAngleWeight = 0.5
        newAngularSpeedWeight = 1.

        # angle should be limited by horizontal speed and position
        targetAngle = state[0]*horPosWeight + state[2]*horSpeedWeight
        if targetAngle >  angleThresh: 
            targetAngle =  angleThresh
        if targetAngle < -angleThresh: 
            targetAngle = -angleThresh

        targetVertPos = np.abs(state[1])*targetVertWeight

        newAngle = (targetAngle - state[4])*newAngleWeight - (state[5])*newAngularSpeedWeight
        newVert = (targetVertPos - state[1])*newVertWeight - (state[3])*newVertWeight

        if state[6] or state[7]:
            newAngle = 0
            newVert = -(state[3])*newVertWeight

        action = g_noEngineFired
        if newAngle > angleMax: 
            action = g_leftEngineFired
        elif newVert > np.abs(newAngle) and newVert > vertMax: 
            action = g_bottomEngineFired
        elif newAngle < -angleMax: 
            action = g_rightEngineFired

        return action

    def getAction(self, environ, state, trial=None):        
        if trial != None:
            if trial < 50:
                action = self.controlsCalc(environ, state)
                return action
        epsilonProbability = self.getEpsilon(trial)
        if np.random.random() > epsilonProbability:
            action = self.getBestAction(state)
        else:
            action = self.action_space.sample()
        return action

    def close(self):
        self.sess.close()
