
import numpy as np
import math, time
import tensorflow as tf

import os.path

import gym
from gym import spaces

## Multilayer perceptron algorithm implementation
## Using tensorflow 
## Input is array of hidden layers
def multilayer_perceptron(x, n_hidden):

    tf.set_random_seed(1)

    totalLayers = len(n_hidden)
    hiddenLayersNum = len(n_hidden) - 1
    inputLayer = x
    currentLayer = inputLayer

    for i in range(1, hiddenLayersNum):
        # Changed from tf.random_normal([n_hidden[i - 1], n_hidden[i]])
        weights = tf.Variable(tf.truncated_normal(          \
            [n_hidden[i - 1], n_hidden[i]],                 \
            stddev= 1. / math.sqrt(float(n_hidden[i - 1]))  \
            ))
        biases = tf.Variable(tf.zeros([n_hidden[i]]))
        nextLayer = tf.nn.tanh(tf.add(tf.matmul(currentLayer, weights), biases))
        currentLayer = nextLayer

    weights = tf.Variable(tf.truncated_normal([n_hidden[totalLayers - 2],       \
                                                n_hidden[totalLayers - 1]],     \
                    stddev=1. / math.sqrt(float(n_hidden[totalLayers - 2])))    \
    )
    biases = tf.Variable(tf.zeros([n_hidden[totalLayers - 1]]))
    outputLayer = tf.matmul(currentLayer, weights) + biases

    return weights, biases, outputLayer

class QLearn(object):

    def __init__(self, observation_space, action_space, reward_range):
        
        self.hiddenLayersNum = 200
        self.batchSize = 100
        self.discountRate = 0.99
        
        self.decayRate = 0.98
        self.epsilonProb = 0.3
        self.altProb = 0.2
        self.updateProbability = 0.2
        self.memSize = 150000

        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_range = reward_range
        np.random.seed(1)

        self.decayLearnRate = 0.99
        self.firstLearnRate = 0.01
        self.stepVal = tf.Variable(0, trainable=False)
        self.learning = tf.train.exponential_decay(self.firstLearnRate, self.stepVal, 100,
                                                    self.decayLearnRate, staircase=True)
        
        self.initQnetwork()

    # def scaleStates(self, states):
    #     if np.isinf(self.observation_space.low).any() or np.isinf(self.observation_space.high).any():
    #         return states
    #     else:
    #         states = (states - self.observation_space.low) / \
    #             (self.observation_space.high - self.observation_space.low) * 2. - 1.
    #         return states * 3

    def getEpsilon(self, episode=None):
        if episode == None:
            return 0.
        else:
            return self.epsilonProb * self.decayRate ** episode

    def initQnetwork(self):

        n_input = self.observation_space.shape[0]
        self.n_out = self.action_space.n

        self.x = tf.placeholder("float", [None, n_input])
        self.y = tf.placeholder("float", [None, 1])
        
        weightsFinal, biasesFinal, self.Q = multilayer_perceptron(self.x, [n_input] + [self.hiddenLayersNum] + [self.n_out])

        self.currentAction = tf.placeholder("float", [None, self.n_out])
        self.singleQ = tf.reduce_sum(self.currentAction * self.Q,
                                     reduction_indices=1)  
        self.singleQ = tf.reshape(self.singleQ, [-1, 1])

        self.errorlist = (self.singleQ - self.y) ** 2

        self.cost = tf.reduce_mean(self.errorlist)
        self.lastcost = 0.
        self.optimizer = tf.train.RMSPropOptimizer(self.learning, 0.9, 0).minimize(self.cost, global_step=self.stepVal)
        self.sess = tf.Session()
        self.pastEvents = []
        self.sess.run(tf.initialize_all_variables())
        self.sess.run(tf.assign(self.stepVal, 0))

    def value(self, states):
        if states.ndim == 1:
            states = states.reshape(1, -1)
        return np.max(self.sess.run(self.Q, feed_dict={self.x: states})).reshape(1, )

    def takeBestAction(self, states):
        if states.ndim == 1:
            states = states.reshape(1, -1)
        return np.argmax(self.sess.run(self.Q, feed_dict={self.x: states}))

    def processMemory(self, index):
        state, action, reward, onew, d, Q, nextState = self.pastEvents[index]

        if Q != None:
            newQ = Q
        else:
            if self.altProb > 0.:
                limitd = 1000
                newQ = reward
                gamma = self.discountRate
                offset = 0
                if nextState == None:
                    newQ += gamma * self.value(onew) * d

                while nextState != None and offset < limitd:
                    offset += nextState
                    n = index + offset
                    newQ += gamma * self.pastEvents[n][2]
                    gamma = gamma * self.discountRate
                    if self.pastEvents[n][6] == None or not (offset < limitd):
                        newQ += gamma * self.value(self.pastEvents[n][3]) * self.pastEvents[n][4]
                    nextState = self.pastEvents[n][6]
            else:
                newQ = 0.
            self.pastEvents[index][5] = newQ
        return state, action, reward, d, onew, newQ

    def learn(self, state, action, newState, reward, finishFlag, nextaction):
        expectedR = reward + self.discountRate * finishFlag * self.value(newState) 
        expectedR = expectedR.reshape(1, )
        possibleStates = state.reshape(1, -1)
        possibleActions = np.array([action])
        possibleRewardsList = expectedR
        
        if (np.random.random() < self.updateProbability) and (len(self.pastEvents) > self.batchSize):
            ind = np.random.choice(len(self.pastEvents), self.batchSize)
            for j in ind:
                state, action, reward, d, onew, newR = self.processMemory(j)
                newR = newR * self.altProb + (reward + self.discountRate \
                    * self.value(onew) * d) * (1. - self.altProb)

                possibleRewardsList = np.concatenate((possibleRewardsList, newR),
                                           0)  
                possibleStates = np.concatenate((possibleStates, state.reshape(1, -1)), 0)
                possibleActions = np.concatenate((possibleActions, np.array([action])), 0)
                
            completeActions = np.zeros((possibleStates.shape[0], self.n_out))
            completeActions[np.arange(possibleActions.shape[0]), possibleActions] = 1.

            self.sess.run(self.optimizer, feed_dict={self.x: possibleStates, self.y: possibleRewardsList.reshape((-1, 1)),
                                                     self.currentAction: completeActions})

        completeActions = np.zeros((possibleStates.shape[0], self.n_out))
        completeActions[np.arange(possibleActions.shape[0]), possibleActions] = 1.
        
        self.sess.run(self.errorlist, feed_dict={self.x: possibleStates, self.y: possibleRewardsList.reshape((-1, 1)),
                                                          self.currentAction: completeActions})
        
        if len(self.pastEvents) > 0 and np.array_equal(self.pastEvents[-1][3], state):
            self.pastEvents[-1][6] = 1
        self.pastEvents.append([state, action, reward, newState, finishFlag, None, None])
        self.pastEvents = self.pastEvents[-self.memSize:]

    def controlsCalc(self, environ, state):
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

    def getAction(self, states, environ, trial=None):
        if trial != None:
            if trial < 50:
                action = self.controlsCalc(environ, states)
                return action

        if np.random.random() > self.getEpsilon(trial):
            action = self.takeBestAction(states)
        else:
            action = self.action_space.sample()
        return action

    def close(self):
        self.sess.close()
