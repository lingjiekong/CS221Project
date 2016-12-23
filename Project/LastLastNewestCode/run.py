
import gym
import numpy as np
import time
import agents
from copy import deepcopy

if __name__ == '__main__':

    numTrials = 400
    seed = 1                           
    environ = gym.make("LunarLander-v2")
    environ.seed(1)
    np.random.seed(1)
    lander = agents.deepQAgent(environ.observation_space, environ.action_space, environ.reward_range)
    
    for trial in range(numTrials):
        
        numTimeSteps = environ.spec.timestep_limit
        totalReward = 0

        state = environ.reset()
        state = lander.scaleStates(state)
        currentState = np.copy(state)

        listOfStates = [currentState]
        listOfActions = []

        for timeStep in range(numTimeSteps):
            action = lander.getAction(environ, currentState, trial)
            (newState, reward, checkFinished, notNeeded) = environ.step(action)

            newState = lander.scaleStates(newState)
            listOfActions.append(action)
            newStateCombined = np.concatenate((currentState[state.shape[0]:], newState))

            listOfStates.append(newStateCombined)

            lander.learn(currentState, action, newStateCombined,  \
                reward, 1.-1.*checkFinished, lander.getAction(environ, newStateCombined, trial))

            currentState = newStateCombined
            totalReward += reward

            if checkFinished:
                break
        listOfStates = np.array(listOfStates)
        listOfActions = np.array(listOfActions)
        allActions = np.zeros((listOfActions.shape[0], lander.numActions))
        allActions[np.arange(listOfActions.shape[0]), listOfActions] = 1.

        print(totalReward)
