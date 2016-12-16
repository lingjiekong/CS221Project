'''
Created on Dec 1, 2016
@author: Stanley Jacob
'''
import numpy as np
import math, time
import tensorflow as tf

import os.path

import gym
from gym import spaces

## Multilayer perceptron algorithm implementation
## Uses tensor flow 
def multilayer_perceptron(_X, numhidden, regulariz, minout=None, maxout=None, initbias=0., outhidden=-1, seed=None):

    tf.set_random_seed(1)

    numlayers = len(numhidden)
    layer_i = _X
    regul = 0.

    for i in range(1, numlayers - 1):
        w = tf.Variable(
            tf.truncated_normal([numhidden[i - 1], numhidden[i]], stddev=1. / math.sqrt(float(numhidden[i - 1]))),
            name="w" + str(i))
        b = tf.Variable(tf.zeros([numhidden[i]]), name="b" + str(i))
        layer_i = tf.nn.sigmoid(tf.add(tf.matmul(layer_i, w), b))
        if outhidden == i:
            hidlayer = layer_i

    w = tf.Variable(tf.truncated_normal([numhidden[numlayers - 2], numhidden[numlayers - 1]], \
                    stddev=1. / math.sqrt(float(numhidden[numlayers - 2]))), \
                    name="w" + str(numlayers - 1))
    b = tf.Variable(tf.zeros([numhidden[numlayers - 1]]), name="b" + str(numlayers - 1))
    layer_out = tf.matmul(layer_i, w) + b + initbias

    if outhidden >= 0:
        return layer_out, regul, hidlayer
    else:
        return layer_out, regul

class deepQAgent(object):

    def __init__(self, observation_space, action_space, reward_range, **userconfig):
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_range = reward_range
        self.config = {
            "memsize": 50000,
            "scalereward": 1.,
            "probupdate": 0.25,
            "lambda": 0.1,
            "past": 0,
            "eps": 0.15,  # Epsilon in epsilon greedy policies
            "decay": 0.996,  # Epsilon decay in epsilon greedy policies
            "initial_learnrate": 0.008,
            "decay_learnrate": 0.999,
            "discount": 0.99,
            "batch_size": 75,
            "hiddenlayers": [400],
            "regularization": [0.0001, 0.000001],
            "momentum": 0.1,
            "file": None,
            "seed": None}
        self.config.update(userconfig)

        if self.config["seed"] is not None:
            np.random.seed(self.config["seed"])
            #print("seed", self.config["seed"])
        self.isdiscrete = isinstance(self.action_space, gym.spaces.Discrete)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.learnrate = tf.train.exponential_decay(self.config['initial_learnrate'], self.global_step, 100,
                                                    self.config['decay_learnrate'], staircase=True, name="learnrate")
        self.bias = tf.Variable(0.0, trainable=False, name="bias")
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
            return self.config['eps'] * self.config['decay'] ** episode

    def incbias(self, l):
        self.sess.run(tf.assign(self.bias, self.bias + l))

    def getbias(self):
        return self.sess.run(self.bias)

    def getlearnrate(self):
        return self.sess.run(self.learnrate)

    ## try comment out
    def stateautoencoder(self, n_input):
        self.plotautoenc = False
        self.outstate, regul, self.hiddencode = multilayer_perceptron(self.sa, [n_input, 2, n_input],
                                                                      [0.000001, 0.000001], outhidden=1,
                                                                      seed=self.config["seed"])
        # print self.x.get_shape(),self.outstate.get_shape()
        self.costautoenc = tf.reduce_mean((self.sa - self.outstate) ** 2) + regul
        self.optimizerautoenc = tf.train.RMSPropOptimizer(self.learnrate).minimize(self.costautoenc,
                                                                                              global_step=self.global_step)

    def initQnetwork(self):
        if self.isdiscrete:
            n_input = self.observation_space.shape[0] * (self.config['past'] + 1)
            self.n_out = self.action_space.n

        self.x = tf.placeholder("float", [None, n_input], name="self.x")
        self.y = tf.placeholder("float", [None, 1], name="self.y")
        self.yR = tf.placeholder("float", [None, 1], name="self.yR")

        self.Qrange = (self.reward_range[0] * 1. / (1. - self.config['discount']),
                       self.reward_range[1] * 1. / (1. - self.config['discount']))
        print(self.Qrange)
        # self.scale=200./max(abs(self.Qrange[1]),abs(self.Qrange[0]))
        self.Q, regul = multilayer_perceptron(self.x, [n_input] + self.config['hiddenlayers'] + [self.n_out],
                                              self.config['regularization'],
                                              initbias=.0, seed=self.config["seed"])  # ,self.Qrange[0],self.Qrange[1])

        self.R, regulR = multilayer_perceptron(self.x, [n_input, 100, self.n_out], self.config['regularization'],
                                               initbias=.0, seed=self.config["seed"])  # ,self.Qrange[0],self.Qrange[1])

        if self.isdiscrete:
            self.curraction = tf.placeholder("float", [None, self.n_out], name="curraction")
            # index = tf.concat(0, [self.out, tf.constant([0, 0, 0], tf.int64)])
            self.singleQ = tf.reduce_sum(self.curraction * self.Q,
                                         reduction_indices=1)  # tf.reduce_sum(input_tensor, reduction_indices, keep_dims, name)#tf.slice(self.Q, [0,self.curraction],[-1,1])  #self.Q[:,self.out]
            self.singleQ = tf.reshape(self.singleQ, [-1, 1])

            self.singleR = tf.reduce_sum(self.curraction * self.R,
                                         reduction_indices=1)  # tf.reduce_sum(input_tensor, reduction_indices, keep_dims, name)#tf.slice(self.Q, [0,self.curraction],[-1,1])  #self.Q[:,self.out]
            self.singleR = tf.reshape(self.singleR, [-1, 1])

            # print 'singleR',self.singleR.get_shape()
            self.errorlistR = (self.singleR - self.yR) ** 2
            self.errorlist = (self.singleQ - self.y) ** 2

        self.cost = tf.reduce_mean(self.errorlist) + regul
        self.costR = tf.reduce_mean(self.errorlistR) + regulR
        self.lastcost = 0.
        self.optimizer = tf.train.RMSPropOptimizer(self.learnrate).minimize(self.cost,
                                                                                                          global_step=self.global_step)
        self.optimizerR = tf.train.RMSPropOptimizer(self.learnrate).minimize(self.costR,
                                                                                      global_step=self.global_step)

        self.sa = tf.placeholder("float", [None, n_input + self.n_out], name="sa")
        self.stateautoencoder(n_input + self.n_out)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.memory = []
        self.errmemory = []
        if self.config['file'] is None or (not os.path.isfile(self.config['file'] + ".tf")):
            self.sess.run(tf.initialize_all_variables())
        else:
            #print("loading " + self.config['file'] + ".tf")
            self.saver.restore(self.sess, self.config['file'] + ".tf")
        self.sess.run(tf.assign(self.global_step, 0))

    def learn(self, state, action, obnew, reward, notdone, nextaction):
        target = reward + self.config['discount'] * self.maxq(obnew) * notdone
        target = target.reshape(1, )
        allstate = state.reshape(1, -1)
        allaction = np.array([action])
        alltarget = target
        indexes = [-1]
        update = (np.random.random() < self.config['probupdate'])
        if update:
            if len(self.memory) > self.config['batch_size']:
                ind = np.random.choice(len(self.memory), self.config[
                    'batch_size'])
                for j in ind:
                    s, a, r, onew, d, Q, nextstate = self.memory[j]

                    if Q != None:
                        alternativetarget = Q
                    else:
                        if self.config['lambda'] > 0.:
                            limitd = 1000
                            alternativetarget = r
                            gamma = self.config['discount']
                            offset = 0
                            if nextstate == None:
                                alternativetarget += gamma * self.maxq(onew) * d

                            while nextstate != None and offset < limitd:
                                offset += nextstate
                                n = j + offset
                                alternativetarget += gamma * self.memory[n][2]
                                gamma = gamma * self.config['discount']
                                if self.memory[n][6] == None or not (offset < limitd):
                                    alternativetarget += gamma * self.maxq(self.memory[n][3]) * self.memory[n][4]
                                nextstate = self.memory[n][6]
                        else:
                            alternativetarget = 0.
                        self.memory[j][5] = alternativetarget

                    alternativetarget = alternativetarget * self.config['lambda'] + (r + self.config[
                        'discount'] * self.maxq(onew) * d) * (1. - self.config['lambda'])

                    alltarget = np.concatenate((alltarget, alternativetarget),
                                               0)  
                    allstate = np.concatenate((allstate, s.reshape(1, -1)), 0)
                    allaction = np.concatenate((allaction, np.array([a])), 0)
                    indexes.append(j)
                allactionsparse = np.zeros((allstate.shape[0], self.n_out))
                allactionsparse[np.arange(allaction.shape[0]), allaction] = 1.

                self.sess.run(self.optimizer, feed_dict={self.x: allstate, self.y: alltarget.reshape((-1, 1)),
                                                         self.curraction: allactionsparse})

                if self.plotautoenc:
                    sa = np.concatenate((allstate, allactionsparse), 1)
                    self.sess.run(self.optimizerautoenc, feed_dict={self.sa: sa, self.outstate: sa})
                
                if False:
                    c = self.sess.run(self.cost, feed_dict={self.x: allstate, self.y: alltarget.reshape((-1, 1)),
                                                            self.curraction: allactionsparse})

                    if c > self.lastcost:
                        self.setlearnrate(0.9999)
                    else:
                        self.setlearnrate(1.0001)
                    self.lastcost = c
        allactionsparse = np.zeros((allstate.shape[0], self.n_out))
        allactionsparse[np.arange(allaction.shape[0]), allaction] = 1.
        indexes = np.array(indexes)
        erlist = self.sess.run(self.errorlist, feed_dict={self.x: allstate, self.y: alltarget.reshape((-1, 1)),
                                                          self.curraction: allactionsparse})
        erlist = erlist.reshape(-1, )
        self.errmemory.append(erlist[0])

        for i, er in enumerate(erlist):
            self.errmemory[indexes[i]] = er
        self.errmemory = self.errmemory[-self.config['memsize']:]

        if len(self.memory) > 0 and np.array_equal(self.memory[-1][3], state):
            self.memory[-1][6] = 1
        self.memory.append([state, action, reward, obnew, notdone, None, None])

        self.memory = self.memory[-self.config['memsize']:]
        return 0

    def maxq(self, observation):
        if self.isdiscrete:
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            return np.max(self.sess.run(self.Q, feed_dict={self.x: observation})).reshape(1, )

    def argmaxq(self, observation):
        # print observation
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        # print observation,self.sess.run(self.Q, feed_dict={self.x:observation})
        return np.argmax(self.sess.run(self.Q, feed_dict={self.x: observation}))

    def heuristic(self, env, s):
        # Heuristic for:
        # 1. Testing. 
        # 2. Demonstration rollout.
        angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
        if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
        if angle_targ < -0.4: angle_targ = -0.4
        hover_targ = 0.2*np.abs(s[1])           # target y should be proporional to horizontal offset

        # PID controller: s[4] angle, s[5] angularSpeed
        angle_todo = (angle_targ - s[4])*0.5 - (s[5])*1.0
        #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

        # PID controller: s[1] vertical coordinate s[3] vertical speed
        hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5
        #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

        if s[6] or s[7]: # legs have contact
            angle_todo = 0
            hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

        if env.continuous:
            a = np.array( [hover_todo*20 - 1, -angle_todo*20] )
            a = np.clip(a, -1, +1)
        else:
            a = 0
            if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2
            elif angle_todo < -0.05: a = 3
            elif angle_todo > +0.05: a = 1
        return a

    def act(self, observation, env, episode=None):
        if episode != None:
            if episode < 50:
                action = self.heuristic(env, observation)
                return action
        eps = self.epsilon(episode)
        if np.random.random() > eps:
            action = self.argmaxq(observation)
        else:
            action = self.action_space.sample()
        return action

    def close(self):
        self.sess.close()

def do_rollout(agent, env, episode, num_steps=None, render=False):
    if num_steps == None:
        num_steps = env.spec.timestep_limit
    total_rew = 0.
    cost = 0.

    ob = env.reset()
    ob = agent.scaleobs(ob)
    ob1 = np.copy(ob)
    for _ in range(agent.config["past"]):
        ob1 = np.concatenate((ob1, ob))
    
    listob = [ob1]
    listact = []

    for t in range(num_steps):

        a = agent.act(ob1, env, episode)

        (obnew, reward, done, _info) = env.step(a)
        obnew = agent.scaleobs(obnew)

        listact.append(a)
        obnew1 = np.concatenate((ob1[ob.shape[0]:], obnew))

        listob.append(obnew1)

        initTime = time.time()
        cost += agent.learn(ob1, a, obnew1, reward, 1. - 1. * done, agent.act(obnew1, env, episode))
        finalTime = time.time()
        learnTime = (finalTime - initTime) * float(100)

        ob1 = obnew1
        total_rew += reward
        # if render and t % 2 == 0:
        # env.render()

        if done: break
    return total_rew, t + 1, cost, listob, listact
