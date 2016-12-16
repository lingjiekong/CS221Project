import numpy as np
import collections # use for sparse vector dot product
import random # use for generate random value initally
import lunarLanderApiMDP
from lunarLanderApiMDP import LunarLanderContinuous
from lunarLanderApiMDP import ContactDetector
from lunarLanderApiMDP import LunarLander
#########################################################################
# define convergence threshold
NUM_STATES = 3*3*3*3*3*3+1
GAMMA = 0.995
TOLERANCE = 0.01
NO_LEARNING_THRESHOLD = 20
LEFT_RIGHT_ACTION = 3
BOTTOM_ACTION = 2

#########################################################################
# define MPD traning parameters
# count time for transition
tranCount = collections.defaultdict(float) 

# real transition probability 
tranProb = collections.defaultdict(float)
for a in range(0,LEFT_RIGHT_ACTION*BOTTOM_ACTION): # 29 different actions include 0
    tranProb[a] = collections.defaultdict(int)
    for startState in range(0, NUM_STATES + 1):
        tranProb[a][startState] = collections.defaultdict(float)
        for endState in range(0, NUM_STATES + 1):
            tranProb[a][startState][endState] = 1.0/float(NUM_STATES)

# count reward time + 1
rewardCountTime = collections.defaultdict(float)
# count reward value + R
rewardCountValue = collections.defaultdict(float)
# real reward used in calcualtion
reward = collections.defaultdict(float)
# real value used in calcualtion and value iteration 
value = collections.defaultdict(float)
for s in range(0,NUM_STATES+1):
    value[s] = 0.1*random.random()
value[729] = 100
#########################################################################
# define the action mapping
actionDict = collections.defaultdict(float)
for r in range(0,BOTTOM_ACTION): # BOTTOM_ACTION = 2
    for lr in range(0,LEFT_RIGHT_ACTION): # LEFT_RIGHT_ACTION = 3
        actionDict[r*3+lr] = [float(r), float(lr)-1]

#########################################################################
# get initial state information of game
env = LunarLanderContinuous()
state = env.reset()

#########################################################################
# learning
consecutiveNoLearningTrials = 0
trainingCount = 0
updateCount = 0
while(trainingCount < 400):
# while(consecutiveNoLearningTrials < NO_LEARNING_THRESHOLD):
    # find what is the best action to take
    maxActionList = list()
    maxActionValue = -float('inf')
    actionValue = 0.0
    stateMap = env.getState(state)
    for a in range(0,LEFT_RIGHT_ACTION*BOTTOM_ACTION):
        for i in list(tranProb[a][stateMap]):
            # tranProb and value is the same every iteration
            actionValue += tranProb[a][stateMap][i] * value[i]
        if actionValue > maxActionValue:
            maxActionValue = actionValue
            maxActionList = list()
            maxActionList.append(a)
        elif actionValue == maxActionValue:
            maxActionList.append(a)
        actionValue = 0
    maxAction = random.choice(maxActionList)

    # get real action from the mapping 
    realAction = np.array(actionDict[maxAction])

    # get the newS/reward/done by simulating the dynamics
    newState, r, done, info = env.step(realAction)
    # env.render()
    newStateMap = env.getState(newState)

    # count for transition probablity 
    if tranCount[maxAction] == 0: 
        tranCount[maxAction] = collections.defaultdict(float) 
        tranCount[maxAction][stateMap] = collections.defaultdict(float)
        tranCount[maxAction][stateMap][newStateMap] = 1    
    elif tranCount[maxAction][stateMap] == 0: 
        tranCount[maxAction][stateMap] = collections.defaultdict(float)
        tranCount[maxAction][stateMap][newStateMap] = 1    
    else: # 
        tranCount[maxAction][stateMap][newStateMap] += 1

    # count for reward 
    rewardCountTime[newStateMap] += 1
    if (newStateMap == NUM_STATES or newStateMap == NUM_STATES - 1):
        rewardCountValue[newStateMap] += (r + (-abs(newState[0])-abs(newState[2])-abs(newState[3])-abs(newState[4])-abs(newState[5])))
    else:
        rewardCountValue[newStateMap] += (-abs(newState[0])-abs(newState[2])-abs(newState[3])-abs(newState[4])-abs(newState[5]))

    # End of one episole of game(win\lose) use value iteration to update policy
    if (newStateMap == NUM_STATES or newStateMap == NUM_STATES - 1):
        # update reward
        for i in list(rewardCountTime):
            reward[i] = float(rewardCountValue[i])/float(rewardCountTime[i])

        # update transition probability
        for a in list(tranCount): # each action
            for i in list(tranCount[a]): # each start state
                totalCount = sum((tranCount[a][i]).values()) # sum total count in start -> end
                DummyTranProb = collections.defaultdict(float)
                for j in list(tranCount[a][i]):
                    DummyTranProb[j] = float(tranCount[a][i][j])/float(totalCount)
                tranProb[a][i] = DummyTranProb

        # value iteration
        updateValue = collections.defaultdict(float)
        updateCount = 0
        vOpt = 0
        err = float('inf')
        while err > TOLERANCE:
            for i in range(0, NUM_STATES + 1):
                vOptMax = -float('inf')
                for a in range(0,LEFT_RIGHT_ACTION*BOTTOM_ACTION):
                    for j in list(tranProb[a][i]):
                        vOpt += tranProb[a][i][j]*value[j]
                    if vOpt > vOptMax:
                        vOptMax = vOpt
                    vOpt = 0
                updateValue[i] = reward[i]+GAMMA*vOptMax
            updateCount += 1
            err = max(abs(updateValue[i] - value[i]) for i in list(value))
            value = updateValue

        # check convergence to stop training
        if updateCount == 1:
            consecutiveNoLearningTrials += 1
        else:
            consecutiveNoLearningTrials = 0
        trainingCount += 1
        print (reward[newStateMap])
        env = LunarLanderContinuous()
        state = env.reset()

#########################################################################  
# Start play the game with the learned T, R, and V
env = LunarLanderContinuous()
# s is the state of the game
s = env.reset()
total_reward = 0
steps = 0
iteration = 10;
# use MDP to play the game
while True:
    for a in range(0,LEFT_RIGHT_ACTION*BOTTOM_ACTION):
        for i in list(tranProb[a][stateMap]):
            actionValue += tranProb[a][stateMap][i] * value[i]
        if actionValue > maxActionValue:
            maxAction = a
        actionValue = 0

    # get real action from the mapping 
    realAction = np.array(actionDict[maxAction])
    # env.step(a) implement action a and render the environment
    s, r, done, info = env.step(realAction)
    env.render()
    total_reward += r
    steps += 1
    if done: break
print ('value')
print (value)
print ('reward')
print (reward)
dummyCount = 0
for a in range(0,LEFT_RIGHT_ACTION*BOTTOM_ACTION): # 29 different actions include 0
    for startState in range(0, NUM_STATES + 1):
        for endState in range(0, NUM_STATES + 1):
            if tranProb[a][startState][endState] !=  0.0013698630136986301:
                dummyCount += 1
print ('dummyCount')
print (dummyCount)
