'''
Created on Dec 1, 2016
@author: Stanley Jacob
@modified: Lingjie Kong
'''
import gym
import logging
import numpy as np
import math, time
import tensorflow as tf
import pickle
from gym import spaces
import os.path

# # Neural Network
# # Use tensorflow library

# class DeepQPlayer(object):
#     def __del__(self):
#         self.close()
#     def __init__(self):

#         # Actions are down, stay still, up
#         ACTIONS_COUNT = 4
#         STATE_FRAMES = 4
#         RESIZED_SCREEN_X, RESIZED_SCREEN_Y = 80, 80


#         self._previous_observations = deque()
#         self._session = tf.Session()
#         self._input_layer, self._output_layer = DeepQPlayer._create_network()()

#         self._action = tf.placeholder("float", [None, self.ACTIONS_COUNT])
#         self._target = tf.placeholder("float", [None])
#         readout_action = tf.reduce_sum(tf.mul(self._output_layer, self._action), reduction_indices=1)
        
#         cost = tf.reduce_mean(tf.square(self._target - readout_action))
#         self._train_operation = tf.train.AdamOptimizer(1e-6).minimize(cost)
        
#         self._session.run(tf.initialize_all_variables())


#     def _train(self):
#         # sample a mini_batch to train on
#         mini_batch = random.sample(self._observations, self.MINI_BATCH_SIZE)
#         # get the batch variables
#         previous_states = [d[self.OBS_LAST_STATE_INDEX] for d in mini_batch]
#         actions = [d[self.OBS_ACTION_INDEX] for d in mini_batch]
#         rewards = [d[self.OBS_REWARD_INDEX] for d in mini_batch]
#         current_states = [d[self.OBS_CURRENT_STATE_INDEX] for d in mini_batch]
#         agents_expected_reward = []
#         # this gives us the agents expected reward for each action we might
#         agents_reward_per_action = self._session.run(self._output_layer, feed_dict={self._input_layer: current_states})
#         for i in range(len(mini_batch)):
#             if mini_batch[i][self.OBS_TERMINAL_INDEX]:
#                 # this was a terminal frame so there is no future reward...
#                 agents_expected_reward.append(rewards[i])
#             else:
#                 agents_expected_reward.append(
#                     rewards[i] + self.FUTURE_REWARD_DISCOUNT * np.max(agents_reward_per_action[i]))

#         # learn that these actions in these states lead to this reward
#         self._session.run(self._train_operation, feed_dict={
#             self._input_layer: previous_states,
#             self._action: actions,
#             self._target: agents_expected_reward})

#         # save checkpoints for later
#         if self._time % self.SAVE_EVERY_X_STEPS == 0:
#             self._saver.save(self._session, self._checkpoint_path + '/network', global_step=self._time)

            
#     @staticmethod
#     def _create_network():
#         # network weights
#         convolution_weights_1 = tf.Variable(tf.truncated_normal([8, 8, self.STATE_FRAMES, 32], stddev=0.01))
#         convolution_bias_1 = tf.Variable(tf.constant(0.01, shape=[32]))

#         convolution_weights_2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
#         convolution_bias_2 = tf.Variable(tf.constant(0.01, shape=[64]))

#         convolution_weights_3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01))
#         convolution_bias_3 = tf.Variable(tf.constant(0.01, shape=[64]))

#         feed_forward_weights_1 = tf.Variable(tf.truncated_normal([256, 256], stddev=0.01))
#         feed_forward_bias_1 = tf.Variable(tf.constant(0.01, shape=[256]))

#         feed_forward_weights_2 = tf.Variable(tf.truncated_normal([256, self.ACTIONS_COUNT], stddev=0.01))
#         feed_forward_bias_2 = tf.Variable(tf.constant(0.01, shape=[self.ACTIONS_COUNT]))

#         input_layer = tf.placeholder("float", [None, self.RESIZED_SCREEN_X, self.RESIZED_SCREEN_Y,
#                                                self.STATE_FRAMES])

#         hidden_convolutional_layer_1 = tf.nn.relu(
#             tf.nn.conv2d(input_layer, convolution_weights_1, strides=[1, 4, 4, 1], padding="SAME") + convolution_bias_1)

#         hidden_max_pooling_layer_1 = tf.nn.max_pool(hidden_convolutional_layer_1, ksize=[1, 2, 2, 1],
#                                                   strides=[1, 2, 2, 1], padding="SAME")

#         hidden_convolutional_layer_2 = tf.nn.relu(
#             tf.nn.conv2d(hidden_max_pooling_layer_1, convolution_weights_2, strides=[1, 2, 2, 1],
#                          padding="SAME") + convolution_bias_2)

#         hidden_max_pooling_layer_2 = tf.nn.max_pool(hidden_convolutional_layer_2, ksize=[1, 2, 2, 1],
#                                                   strides=[1, 2, 2, 1], padding="SAME")

#         hidden_convolutional_layer_3 = tf.nn.relu(
#             tf.nn.conv2d(hidden_max_pooling_layer_2, convolution_weights_3,
#                          strides=[1, 1, 1, 1], padding="SAME") + convolution_bias_3)

#         hidden_max_pooling_layer_3 = tf.nn.max_pool(hidden_convolutional_layer_3, ksize=[1, 2, 2, 1],
#                                                   strides=[1, 2, 2, 1], padding="SAME")

#         hidden_convolutional_layer_3_flat = tf.reshape(hidden_max_pooling_layer_3, [-1, 256])

#         final_hidden_activations = tf.nn.relu(
#             tf.matmul(hidden_convolutional_layer_3_flat, feed_forward_weights_1) + feed_forward_bias_1)

#         output_layer = tf.matmul(final_hidden_activations, feed_forward_weights_2) + feed_forward_bias_2

#         return input_layer, output_layer

#     def _choose_next_action(self):
#     new_action = np.zeros([self.ACTIONS_COUNT])

#     if (not self._playback_mode) and (random.random() <= self._probability_of_random_action):
#         # choose an action randomly
#         action_index = random.randrange(self.ACTIONS_COUNT)
#     else:
#         # choose an action given our last state
#         readout_t = self._session.run(self._output_layer, feed_dict={self._input_layer: [self._last_state]})[0]
#         if self.verbose_logging:
#             print("Action Q-Values are %s" % readout_t)
#         action_index = np.argmax(readout_t)

#     new_action[action_index] = 1
#     return new_action

#     @staticmethod
#     def _key_presses_from_action(action_set):
#         if action_set[0] == 1:
#             return [K_DOWN]
#         elif action_set[1] == 1:
#             return []
#         elif action_set[2] == 1:
#             return [K_UP]
#         raise Exception("Unexpected action")



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
