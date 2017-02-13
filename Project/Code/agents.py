'''
Created on Dec 1, 2016
@author: Lingjie Kong, Stanley Jacobs
'''
import gym
import logging
import numpy as np
import math, time
import tensorflow as tf
import pickle
from gym import spaces
import os.path


##########################################################################
## Multilayer perceptron algorithm implementation
## Uses tensor flow 
def mlp(inputLayer, n_hidden, regParam):
    tf.set_random_seed(1)
    totalLayers = len(n_hidden)
    currentLayer = inputLayer
    regulTerm = 0.
    for i in range(1, totalLayers - 1):
        weights = tf.Variable(
            tf.truncated_normal(                \
                [n_hidden[i - 1], n_hidden[i]], \
                stddev=1. / math.sqrt(float(n_hidden[i - 1]))))
        biases = tf.Variable(tf.zeros([n_hidden[i]]))
        nextLayer = tf.nn.tanh(tf.add(tf.matmul(currentLayer, weights), biases))
        currentLayer = nextLayer
        regulTerm += tf.nn.l2_loss(weights) * regParam[i - 1]
        
    weights = tf.Variable(tf.truncated_normal([n_hidden[totalLayers - 2], \
                    n_hidden[totalLayers - 1]], \
                    stddev=1. / math.sqrt(float(n_hidden[totalLayers - 2]))))

    biases = tf.Variable(tf.zeros([n_hidden[totalLayers - 1]]))

    outputLayer = tf.matmul(currentLayer, weights) + biases
    
    regulTerm += tf.nn.l2_loss(weights) * regParam[totalLayers - 2]

    return outputLayer, regulTerm



class deepQAgent(object):
    def __del__(self):
        self.close()

    def __init__(self, observation_space, action_space, reward_range):
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_range = reward_range
        
        self.discountRate = 0.99

        self.startLearnRate = 0.008
        self.learnDecay = 0.999

        self.epsilonProb = 0.15
        self.epsilonDecay = 0.996

        self.lambdaProb = 0.1
        self.updateProb = 0.25

        self.batchSize = 75
        self.memSize = 50000

        self.hiddenNum = 300
        self.regulParam = [0.0001, 0.000001]

        np.random.seed(1)

        self.global_step = tf.Variable(0, trainable=False)
        self.learnrate = tf.train.exponential_decay(self.startLearnRate, self.global_step, 100,
                                                    self.learnDecay, staircase=True)
        
        self.initQnetwork()

    def scaleobs(self, obs):
        if np.isinf(self.observation_space.low).any() or np.isinf(self.observation_space.high).any():
            return obs
        else:
            o = (obs - self.observation_space.low) / (
                self.observation_space.high - self.observation_space.low) * 2. - 1.
            return o * 3

    def epsilon(self, episode=None):
        if episode == None:
            return 0.
        else:
            return self.epsilonProb * self.epsilonDecay ** episode

    def initQnetwork(self):
        n_input = self.observation_space.shape[0]
        self.n_out = self.action_space.n

        self.x = tf.placeholder("float", [None, n_input])
        self.y = tf.placeholder("float", [None, 1])
        
        self.Q, regTerm = mlp(self.x, [n_input] + [self.hiddenNum] + [self.n_out],
                                              self.regulParam)

        self.currentAction = tf.placeholder("float", [None, self.n_out])
        
        self.singleQ = tf.reduce_sum(self.currentAction * self.Q,
                                     reduction_indices=1)  
        self.singleQ = tf.reshape(self.singleQ, [-1, 1])

        self.lossFull = (self.singleQ - self.y) ** 2

        self.cost = tf.reduce_mean(self.lossFull) + regTerm
        
        self.optimizer = tf.train.RMSPropOptimizer(self.learnrate, 0.9, 0.1).minimize(self.cost, \
            global_step=self.global_step)
        self.sess = tf.Session()
        self.pastActions = []
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.assign(self.global_step, 0))

    def updateMemory(self, index):
        state, action, reward, onew, d, Q, nextstate = self.pastActions[index]

        if Q != None:
            alternativetarget = Q
        else:
            if self.lambdaProb > 0.:
                limitd = 1000
                alternativetarget = reward
                gamma = self.discountRate
                offset = 0
                if nextstate == None:
                    alternativetarget += gamma * self.maxq(onew) * d
                
                while nextstate != None and offset < limitd:
                    offset += nextstate
                    n = index + offset

                    alternativetarget += gamma * self.pastActions[n][2]
                    gamma = gamma * self.discountRate
                    if self.pastActions[n][6] == None or not (offset < limitd):
                        alternativetarget += gamma * self.maxq(self.pastActions[n][3]) * self.pastActions[n][4]
                    nextstate = self.pastActions[n][6]
            else:
                alternativetarget = 0.
            self.pastActions[index][5] = alternativetarget
        
        alternativetarget = alternativetarget * self.lambdaProb + (reward + self.discountRate \
            * self.maxq(onew) * d) * (1. - self.lambdaProb)

        return alternativetarget, state, action

    def learn(self, state, action, obnew, reward, notdone, nextaction):
        target = reward + self.discountRate * self.maxq(obnew) * notdone
        
        target = target.reshape(1, )
        allstate = state.reshape(1, -1)
        allaction = np.array([action])
        alltarget = target
        
        update = (np.random.random() < self.updateProb)
        if update:
            if len(self.pastActions) > self.batchSize:
                ind = np.random.choice(len(self.pastActions), self.batchSize) 
                for j in ind:
                    alternativetarget, someState, someAction = self.updateMemory(j)
                    alltarget = np.concatenate((alltarget, alternativetarget), 0)  
                    allstate = np.concatenate((allstate, someState.reshape(1, -1)), 0)
                    allaction = np.concatenate((allaction, np.array([someAction])), 0)
                    
                allactionsparse = np.zeros((allstate.shape[0], self.n_out))
                allactionsparse[np.arange(allaction.shape[0]), allaction] = 1.

                self.sess.run(self.optimizer, feed_dict={self.x: allstate, self.y: alltarget.reshape((-1, 1)),
                                                         self.currentAction: allactionsparse})

        allactionsparse = np.zeros((allstate.shape[0], self.n_out))
        allactionsparse[np.arange(allaction.shape[0]), allaction] = 1.
        
        self.sess.run(self.lossFull, feed_dict={self.x: allstate, self.y: alltarget.reshape((-1, 1)),
                                                          self.currentAction: allactionsparse})
        
        if len(self.pastActions) > 0 and np.array_equal(self.pastActions[-1][3], state):
            self.pastActions[-1][6] = 1
        self.pastActions.append([state, action, reward, obnew, notdone, None, None])
        
        self.pastActions = self.pastActions[-self.memSize:]
        return 0

    def maxq(self, observation):
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        return np.max(self.sess.run(self.Q, feed_dict={self.x: observation})).reshape(1, )

    def argmaxq(self, observation):
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        return np.argmax(self.sess.run(self.Q, feed_dict={self.x: observation}))

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

    def act(self, env, observation, episode=None):
        
        if episode != None:
            if episode < 50:
                action = self.controlsCalc(env, observation)
                return action

        eps = self.epsilon(episode)

        # epsilon greedy.
        if np.random.random() > eps:
            action = self.argmaxq(observation)
        else:
            action = self.action_space.sample()
        return action

    def close(self):
        self.sess.close()
